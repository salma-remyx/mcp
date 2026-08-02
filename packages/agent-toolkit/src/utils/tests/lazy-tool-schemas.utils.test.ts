import { Tool, ToolType } from '../../core/tool';
import {
  scoreToolRelevance,
  selectRelevantToolNames,
  summarizeJsonSchema,
  tokenizeText,
} from '../tools/lazy-tool-schemas.utils';

const makeTool = (name: string, description: string, argNames: string[] = []): Tool<any, any> =>
  ({
    name,
    type: ToolType.READ,
    annotations: { audience: [] },
    getDescription: () => description,
    getInputSchema: () => Object.fromEntries(argNames.map((key) => [key, {}])),
    execute: jest.fn(),
  }) as unknown as Tool<any, any>;

describe('lazy-tool-schemas.utils', () => {
  describe('tokenizeText', () => {
    it('lowercases, splits separators, and dedupes', () => {
      expect(tokenizeText('Create_Column create-column Create.Column')).toEqual(['create', 'column']);
    });

    it('drops stopwords and single characters', () => {
      expect(tokenizeText('I want to use a board')).toEqual(['board']);
    });

    it('returns an empty array for blank input', () => {
      expect(tokenizeText(undefined)).toEqual([]);
      expect(tokenizeText('   ')).toEqual([]);
    });
  });

  describe('summarizeJsonSchema', () => {
    const fullSchema = {
      $schema: 'http://json-schema.org/draft-07/schema#',
      type: 'object',
      additionalProperties: false,
      properties: {
        color: { type: 'string', enum: ['red', 'green', 'blue'], description: 'Pick a color' },
        count: { type: 'number', description: 'How many' },
      },
      required: ['color'],
    };

    it('keeps structure (type/properties/required) but drops prose, enums, and bookkeeping keys', () => {
      const summary = summarizeJsonSchema(fullSchema);

      expect(summary).toEqual({
        type: 'object',
        properties: {
          color: { type: 'string' },
          count: { type: 'number' },
        },
        required: ['color'],
      });
    });

    it('does not mutate the input schema', () => {
      const snapshot = JSON.parse(JSON.stringify(fullSchema));
      summarizeJsonSchema(fullSchema);
      expect(fullSchema).toEqual(snapshot);
    });

    it('passes through non-object nodes unchanged', () => {
      expect(summarizeJsonSchema('hello')).toBe('hello');
      expect(summarizeJsonSchema(42)).toBe(42);
      expect(summarizeJsonSchema(null)).toBeNull();
    });
  });

  describe('scoreToolRelevance', () => {
    it('scores name + description + arg-name overlap against the query', () => {
      const tool = makeTool('create_column', 'Create a new column in a board', ['columnTitle', 'columnType']);
      // query tokens: {create, column}; tool tokens include both -> score 2
      expect(scoreToolRelevance(tool, 'create a column')).toBe(2);
    });

    it('returns 0 when the query shares no tokens with the tool', () => {
      const tool = makeTool('list_boards', 'List boards in a workspace', ['limit']);
      expect(scoreToolRelevance(tool, 'create a column')).toBe(0);
    });

    it('returns 0 for an empty query', () => {
      const tool = makeTool('create_column', 'Create a column', ['columnTitle']);
      expect(scoreToolRelevance(tool, '')).toBe(0);
    });
  });

  describe('selectRelevantToolNames', () => {
    // Controlled fixtures so token overlap is unambiguous.
    const boardAlpha = makeTool('board_alpha', 'alpha board', []);
    const boardBeta = makeTool('board_beta', 'beta board', []);
    const gamma = makeTool('gamma_tool', 'gamma something else', []);

    it('promotes only the positively-relevant tool when the cap is 1', () => {
      const names = selectRelevantToolNames([boardAlpha, boardBeta, gamma], 'alpha', 1);
      expect(names.size).toBe(1);
      expect(names.has('board_alpha')).toBe(true);
      expect(names.has('board_beta')).toBe(false);
    });

    it('promotes every positively-relevant tool when under the cap', () => {
      // 2 tools match 'board'; cap 2 leaves no room for the non-matching filler.
      const names = selectRelevantToolNames([boardAlpha, boardBeta, gamma], 'board', 2);
      expect(names.has('board_alpha')).toBe(true);
      expect(names.has('board_beta')).toBe(true);
      expect(names.has('gamma_tool')).toBe(false);
    });

    it('never promotes more than the cap even when many tools are relevant', () => {
      // Both board tools match 'board'; cap 1 keeps only one (tie broken by index).
      const names = selectRelevantToolNames([boardAlpha, boardBeta, gamma], 'board', 1);
      expect(names.size).toBe(1);
      expect(names.has('board_alpha')).toBe(true);
    });

    it('falls back to original order to fill the cap when nothing matches', () => {
      const names = selectRelevantToolNames([boardAlpha, boardBeta, gamma], 'zzz', 2);
      expect(names.size).toBe(2);
      expect(Array.from(names)).toEqual(['board_alpha', 'board_beta']);
    });
  });
});
