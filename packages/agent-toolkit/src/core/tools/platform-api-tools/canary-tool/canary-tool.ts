import { z } from 'zod';
import { ToolInputType, ToolOutputType, ToolType } from '../../../tool';
import { BaseMondayApiTool, createMondayApiAnnotations } from '../base-monday-api-tool';

/**
 * Canary tools are diagnostic probe tools planted in an agent's tool set to
 * surface tool-selection weaknesses. Each canary mimics the surface of a real
 * tool (name plus description) but performs no real action: when an agent
 * selects and calls it, execute() returns a structured diagnostic that names
 * the trap type and points at the legitimate tool that should have been used.
 *
 * Adapted from "Diagnosing Tool-Selection Reasoning in LLM Agents with Canary
 * Tools" (arXiv 2608.04719). That paper's six-type taxonomy is encoded in
 * CanaryTrapType below. This module ships the canary mechanism plus two
 * concrete probes covering two distinct taxonomy dimensions
 * (semantic_decoy and capability_mirage). The paper's full evaluation
 * harness (its benchmark of 120 tasks, canary-density conditions, the
 * provider-independent judge, and per-model susceptibility reporting) is
 * intentionally out of scope here: evaluation belongs in a downstream PR.
 *
 * Canaries are disabled by default (enabledByDefault = false) so they never
 * enter a normal agent's tool set. Enable one explicitly through
 * ToolsConfiguration.include when running a diagnostic.
 */

export const CanaryTrapType = {
  SEMANTIC_DECOY: 'semantic_decoy',
  PARAMETER_TRAP: 'parameter_trap',
  CAPABILITY_MIRAGE: 'capability_mirage',
  PREREQUISITE_BLINDNESS: 'prerequisite_blindness',
  TEMPORAL_DECOY: 'temporal_decoy',
  GRANULARITY_TRAP: 'granularity_trap',
} as const;

export type CanaryTrapTypeValue = (typeof CanaryTrapType)[keyof typeof CanaryTrapType];

/**
 * Structured result returned when a canary fires. trap_type is the profiling
 * dimension from the paper's taxonomy: collecting trap_type across many fired
 * canaries turns a single "wrong tool" outcome into a multi-dimensional profile
 * of how an agent reasons about tools.
 */
export interface CanaryDiagnostic {
  canary_triggered: boolean;
  canary_tool: string;
  trap_type: CanaryTrapTypeValue;
  message: string;
  explanation: string;
  correct_tool: string;
  guidance: string;
}

const canaryToolSchema = {
  doc_id: z.string().optional().describe('Identifier of the document to inspect, as returned by read_docs.'),
  query: z.string().optional().describe('What you want to find or ask about the document.'),
};

/**
 * Base class for canary probes. Subclasses supply the lure surface
 * (name, lureDescription) and the diagnostic metadata (trapType, correctTool,
 * trapExplanation). executeInternal performs no API call: a canary's only job
 * is to be selectable, and once selected, to report that it was selected.
 */
export abstract class BaseCanaryTool extends BaseMondayApiTool<typeof canaryToolSchema> {
  abstract readonly trapType: CanaryTrapTypeValue;
  abstract readonly correctTool: string;
  protected abstract readonly lureDescription: string;
  protected abstract readonly trapExplanation: string;

  // Diagnostic probes must not silently appear in every agent's tool set.
  enabledByDefault = false;

  type = ToolType.READ;

  annotations = createMondayApiAnnotations({
    title: 'Canary Diagnostic Probe',
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
  });

  getDescription(): string {
    return this.lureDescription;
  }

  getInputSchema(): typeof canaryToolSchema {
    return canaryToolSchema;
  }

  protected async executeInternal(
    _input?: ToolInputType<typeof canaryToolSchema>,
  ): Promise<ToolOutputType<never>> {
    return {
      content: {
        canary_triggered: true,
        canary_tool: this.name,
        trap_type: this.trapType,
        message:
          'You called a canary tool. Canaries are diagnostic probes planted to surface tool-selection weaknesses and perform no real action.',
        explanation: this.trapExplanation,
        correct_tool: this.correctTool,
        guidance: `Re-issue the request using ${this.correctTool}, the legitimate tool for this task.`,
      },
    };
  }
}

/**
 * Semantic-decoy canary. Its name and description closely mimic read_docs,
 * luring selection by surface similarity. Selecting it indicates the choice was
 * driven by name resemblance rather than the documented behavior of the real
 * tool. The legitimate tool is read_docs.
 */
export class ReadDocSummaryCanaryTool extends BaseCanaryTool {
  name = 'read_doc_summary';
  trapType = CanaryTrapType.SEMANTIC_DECOY;
  correctTool = 'read_docs';

  protected readonly lureDescription = `Read a concise summary of a monday.com document.

Provide a doc_id to get a short plain-text summary of the document content. Use this when you only need the gist of a document and do not want the full content.`;

  protected readonly trapExplanation = `read_doc_summary mimics read_docs by name and description but performs no read at all. Selecting it suggests the tool was chosen by surface name similarity rather than its documented behavior.`;
}

/**
 * Capability-mirage canary. Its description claims a question-answering
 * capability that no monday.com document tool actually provides: read_docs
 * returns content for the agent to reason over, it does not answer questions.
 * The paper found capability mirages most reliably trap capable models, so this
 * probe targets that dimension. The legitimate tool is read_docs.
 */
export class AnswerDocQuestionsCanaryTool extends BaseCanaryTool {
  name = 'answer_doc_questions';
  trapType = CanaryTrapType.CAPABILITY_MIRAGE;
  correctTool = 'read_docs';

  protected readonly lureDescription = `Answer natural-language questions about a monday.com document.

Give a doc_id and a question to receive a direct answer drawn from the document content.`;

  protected readonly trapExplanation = `answer_doc_questions advertises a question-answering capability that the platform document tools do not provide. read_docs returns content to reason over rather than answering questions. Selecting it reveals an inferred capability that was never verified against the tool behavior.`;
}
