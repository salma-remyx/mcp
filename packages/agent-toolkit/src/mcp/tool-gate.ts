/**
 * Query-driven tool gate — reduces the per-turn "MCP/Tools Tax".
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy
 * Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic
 * Workflows" (arXiv:2604.21816). That paper's core mechanism is a
 * query-conditioned gate that exposes only the tools relevant to the current
 * turn, shrinking the tool-schema payload injected into the context each turn.
 *
 * The paper estimates per-turn relevance with a learned attention model. This
 * module substitutes that learned estimator with a parameter-free lexical proxy
 * (query/tool token overlap weighted by corpus IDF), which approximates the same
 * "which tools match this query" signal without any training data or weights.
 * The gate is applied through the repo's existing enable/disable contract:
 * disabled tools are not advertised to the client, so the active schema set
 * becomes query-adaptive — the same effect the paper calls "lazy schema
 * loading", delivered natively rather than via a separate on-demand loader.
 */

/** Minimal descriptor the gate needs for a registered tool. */
export interface ToolDescriptor {
  name: string;
  description: string;
}

/** Options controlling how aggressively the gate prunes the tool set. */
export interface ToolGateOptions {
  /** Max number of relevance-selected tools to keep enabled (default 15). */
  topK?: number;
  /** Minimum normalized relevance in [0, 1] for a tool to be kept (default 0.03). */
  minScore?: number;
  /** Tool names kept enabled regardless of score, e.g. the discovery tool. */
  alwaysOn?: string[];
}

/** Outcome of a gate pass: which tools to enable vs. disable. */
export interface ToolGateResult {
  enabled: string[];
  disabled: string[];
  /** Per-tool normalized relevance scores, for observability/debugging. */
  scores: Record<string, number>;
}

const DEFAULT_TOP_K = 15;
const DEFAULT_MIN_SCORE = 0.03;
/** Token-frequency multiplier for tool-name hits — name matches are a strong signal. */
const NAME_WEIGHT = 3;

// Generic articles / conjunctions / prepositions / pronouns. Domain verbs
// (create, get, list, search, ...) are intentionally NOT included — they carry
// the tool-selection signal. Packed in one string so the literal stays compact.
const STOPWORDS = new Set(
  (
    'the a an and or not to of for in on at by with from as is are be been being was were ' +
    'this that these those it its i me my we our you your he she they them their how what ' +
    'which who whom do does did can could would should will shall may might please want need'
  ).split(' '),
);

/**
 * Lowercase, split on non-alphanumerics (handles snake_case names), drop short
 * tokens and generic stopwords. Returns raw tokens — no stemming — to keep the
 * gate deterministic and dependency-free.
 */
export function tokenize(text: string): string[] {
  if (!text) return [];
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 2 && !STOPWORDS.has(token));
}

interface CorpusStats {
  /** Smoothed inverse document frequency per token, computed over tool docs. */
  idf: Map<string, number>;
  /** Term frequency per tool, with name tokens counted NAME_WEIGHT times. */
  toolTf: Map<string, Map<string, number>>;
}

/** Build IDF and per-tool TF tables from the descriptor corpus. */
function buildCorpus(descriptors: ToolDescriptor[]): CorpusStats {
  const docCount = descriptors.length;
  const df = new Map<string, number>();
  const toolTf = new Map<string, Map<string, number>>();

  for (const tool of descriptors) {
    const tf = new Map<string, number>();
    const add = (token: string, weight: number) => {
      tf.set(token, (tf.get(token) ?? 0) + weight);
    };

    const nameTokens = tokenize(tool.name);
    const seenForDf = new Set<string>();
    for (const token of nameTokens) {
      add(token, NAME_WEIGHT);
      seenForDf.add(token);
    }
    for (const token of tokenize(tool.description)) {
      add(token, 1);
      seenForDf.add(token);
    }
    toolTf.set(tool.name, tf);
    for (const token of seenForDf) {
      df.set(token, (df.get(token) ?? 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  for (const [token, freq] of df) {
    // Smoothed IDF (sklearn-style): keeps every token's weight positive.
    idf.set(token, Math.log((docCount + 1) / (freq + 1)) + 1);
  }

  return { idf, toolTf };
}

/**
 * Score every tool against the query. The score is the fraction of the query's
 * IDF "mass" that lands in each tool's document — a [0, 1] relevance where 1
 * means every query term is accounted for (with multiplicity) in that tool.
 */
export function scoreToolsForQuery(
  query: string,
  descriptors: ToolDescriptor[],
): Array<{ name: string; score: number }> {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0 || descriptors.length === 0) {
    return descriptors.map((d) => ({ name: d.name, score: 0 }));
  }

  const { idf, toolTf } = buildCorpus(descriptors);

  // Distinct query terms with their cumulative IDF weight (denominator).
  const queryWeight = new Map<string, number>();
  for (const token of queryTokens) {
    queryWeight.set(token, (queryWeight.get(token) ?? 0) + (idf.get(token) ?? 1));
  }
  const totalQueryWeight = [...queryWeight.values()].reduce((sum, w) => sum + w, 0);
  if (totalQueryWeight <= 0) {
    return descriptors.map((d) => ({ name: d.name, score: 0 }));
  }

  return descriptors.map((tool) => {
    const tf = toolTf.get(tool.name);
    if (!tf) return { name: tool.name, score: 0 };

    let matched = 0;
    for (const [token, qWeight] of queryWeight) {
      const termFreq = tf.get(token) ?? 0;
      if (termFreq > 0) {
        matched += qWeight * termFreq;
      }
    }
    // Relevance = fraction of the query's IDF mass that lands in this tool's
    // document. Name hits count NAME_WEIGHT times, so name matches dominate.
    // Clamped to [0, 1]; a query term repeated in the doc can push raw above 1.
    const raw = matched / totalQueryWeight;
    return { name: tool.name, score: Math.min(1, raw) };
  });
}

/**
 * Decide which tools to keep active for a query. Tools above `minScore` are
 * ranked and the top `topK` are kept; `alwaysOn` tools are kept unconditionally.
 *
 * Safety: if no tool overlaps the query at all (opaque/empty query), nothing is
 * pruned — the gate refuses to gut the tool set when it has no signal.
 */
export function selectRelevantTools(
  query: string,
  descriptors: ToolDescriptor[],
  options?: ToolGateOptions,
): ToolGateResult {
  const { topK = DEFAULT_TOP_K, minScore = DEFAULT_MIN_SCORE, alwaysOn = [] } = options ?? {};

  const scored = scoreToolsForQuery(query, descriptors);
  const scores: Record<string, number> = {};
  scored.forEach((s) => {
    scores[s.name] = s.score;
  });

  const maxScore = scored.reduce((max, s) => Math.max(max, s.score), 0);
  // No lexical signal → keep the full set active rather than disabling everything.
  if (maxScore <= 0) {
    return {
      enabled: descriptors.map((d) => d.name),
      disabled: [],
      scores,
    };
  }

  const ranked = [...scored].sort((a, b) => b.score - a.score);
  const candidates = ranked.filter((s) => s.score >= minScore).map((s) => s.name);
  const selected = new Set<string>([...candidates.slice(0, topK), ...alwaysOn]);

  const enabled: string[] = [];
  const disabled: string[] = [];
  for (const descriptor of descriptors) {
    if (selected.has(descriptor.name)) {
      enabled.push(descriptor.name);
    } else {
      disabled.push(descriptor.name);
    }
  }

  return { enabled, disabled, scores };
}
