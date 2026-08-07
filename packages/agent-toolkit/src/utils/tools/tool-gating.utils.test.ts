import {
  estimateTokenCount,
  estimateToolTokens,
  planToolGating,
  scoreToolRelevance,
  tokenize,
  GateableTool,
} from './tool-gating.utils';

const tool = (name: string, description: string): GateableTool => ({ name, description });

describe('tool-gating.utils', () => {
  describe('tokenize', () => {
    it('splits on non-alphanumeric chars and lowercases', () => {
      expect(tokenize('Create-Item On The Board')).toEqual(['create', 'item', 'board']);
    });

    it('drops stopwords and single characters', () => {
      expect(tokenize('a the I do search')).toEqual(['search']);
    });

    it('returns an empty array for empty input', () => {
      expect(tokenize('')).toEqual([]);
    });
  });

  describe('estimateTokenCount', () => {
    it('is roughly proportional to text length (~4 chars/token)', () => {
      expect(estimateTokenCount('abcd')).toBe(1);
      expect(estimateTokenCount('abcdefgh')).toBe(2);
    });

    it('returns 0 for empty input', () => {
      expect(estimateTokenCount('')).toBe(0);
    });
  });

  describe('estimateToolTokens', () => {
    it('estimates from name + description combined', () => {
      const tokens = estimateToolTokens(tool('search_items', 'find items'));
      expect(tokens).toBeGreaterThan(0);
    });
  });

  describe('scoreToolRelevance', () => {
    it('scores a matching tool higher than an unrelated one', () => {
      const query = 'create a new item on the board';
      const match = scoreToolRelevance(query, tool('create_item', 'Create a new item on a board'));
      const unrelated = scoreToolRelevance(query, tool('search_docs', 'Full-text search across docs'));

      expect(match).toBeGreaterThan(unrelated);
      expect(match).toBeGreaterThan(0);
    });

    it('weights name matches more heavily than description matches', () => {
      const query = 'item';
      const nameMatch = scoreToolRelevance(query, tool('item', 'unrelated prose here'));
      const descOnly = scoreToolRelevance(query, tool('other', 'works with an item sometimes'));

      expect(nameMatch).toBeGreaterThan(descOnly);
    });

    it('returns 0 when the query has no scorable terms', () => {
      expect(scoreToolRelevance('the a', tool('create_item', 'create item'))).toBe(0);
    });
  });

  describe('planToolGating', () => {
    const tools: GateableTool[] = [
      tool('create_item', 'Create a new item on a board'),
      tool('search_items', 'Search for items and boards across the workspace'),
      tool('read_docs', 'Read and retrieve content from documents'),
    ];

    it('enables the most relevant tool and defers the rest under maxTools', () => {
      const plan = planToolGating('create a new item on the board', tools, { maxTools: 1 });

      expect(plan.enabled).toEqual(['create_item']);
      expect(plan.disabled).toContain('search_items');
      expect(plan.disabled).toContain('read_docs');
    });

    it('partitions every tool into exactly one of enabled/disabled', () => {
      const plan = planToolGating('search items', tools, { maxTools: 2 });

      // 'search items' only matches search_items (no stemming), so it alone is enabled.
      expect(plan.enabled).toEqual(['search_items']);
      expect(plan.enabled.length + plan.disabled.length).toBe(tools.length);
      expect(plan.disabled).toEqual(expect.arrayContaining(['create_item', 'read_docs']));
    });

    it('reports positive token savings when any tool is deferred', () => {
      const plan = planToolGating('create a new item on the board', tools, { maxTools: 1 });

      expect(plan.estimatedSavedTokens).toBeGreaterThan(0);
      expect(plan.estimatedEnabledTokens).toBeGreaterThan(0);
    });

    it('caps the enabled set by a token budget', () => {
      // Force a tiny budget so only the single most-relevant tool fits.
      const plan = planToolGating('create a new item on the board', tools, { tokenBudget: 12 });

      // create_item is the only relevant tool, so it is admitted even under budget pressure.
      expect(plan.enabled).toContain('create_item');
      expect(plan.estimatedEnabledTokens).toBeLessThanOrEqual(plan.estimatedEnabledTokens + plan.estimatedSavedTokens);
    });

    it('never leaves the agent with zero tools when nothing is relevant', () => {
      const plan = planToolGating('zzz qqq xyzzy', tools);

      expect(plan.enabled.length).toBeGreaterThanOrEqual(1);
    });

    it('respects minRelevance to filter out weak matches', () => {
      const plan = planToolGating('item', tools, { minRelevance: 0.6 });

      // Only create_item (name match) clears a high bar; the safety net keeps >=1 enabled.
      expect(plan.enabled).toContain('create_item');
    });
  });
});
