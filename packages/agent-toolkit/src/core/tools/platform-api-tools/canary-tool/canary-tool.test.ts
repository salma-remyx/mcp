import { allGraphqlApiTools } from '..';
import { BaseMondayApiTool, BaseMondayApiToolConstructor } from '../base-monday-api-tool';
import { createMockApiClient } from '../test-utils/mock-api-client';
import {
  AnswerDocQuestionsCanaryTool,
  CanaryTrapType,
  ReadDocSummaryCanaryTool,
} from './canary-tool';

describe('Canary diagnostic tools', () => {
  const { mockApiClient, mockRequest } = createMockApiClient();

  // Typed as the repo's own constructor alias so the registry-membership and
  // instantiation checks mirror tool-description-safety.test.ts exactly.
  const canaryCtors: BaseMondayApiToolConstructor[] = [
    ReadDocSummaryCanaryTool,
    AnswerDocQuestionsCanaryTool,
  ];

  describe('registration through allGraphqlApiTools', () => {
    it.each(canaryCtors)(
      '%s is registered and instantiates via the base contract',
      (Ctor: BaseMondayApiToolConstructor) => {
        // Exercises the call-site edit in platform-api-tools/index.ts: each canary
        // must live in the same registry as every other tool and construct via the
        // real BaseMondayApiTool constructor contract.
        expect(allGraphqlApiTools).toContain(Ctor);
        const tool = new Ctor(mockApiClient);
        expect(tool).toBeInstanceOf(BaseMondayApiTool);
        // Diagnostic probes must stay out of the default tool set.
        expect(tool.enabledByDefault).toBe(false);
      },
    );
  });

  describe('ReadDocSummaryCanaryTool (semantic_decoy)', () => {
    it('returns a typed diagnostic and makes no API call when invoked', async () => {
      const tool = new ReadDocSummaryCanaryTool(mockApiClient);
      // Runs through BaseMondayApiTool.execute(), the real public entry point.
      const result = await tool.execute({ doc_id: '123', query: 'summary' });

      expect(result.content).toEqual({
        canary_triggered: true,
        canary_tool: 'read_doc_summary',
        trap_type: CanaryTrapType.SEMANTIC_DECOY,
        message: expect.any(String),
        explanation: expect.any(String),
        correct_tool: 'read_docs',
        guidance: expect.stringContaining('read_docs'),
      });
      // A canary is a probe, not a real read: it must never hit the API.
      expect(mockRequest).not.toHaveBeenCalled();
    });
  });

  describe('AnswerDocQuestionsCanaryTool (capability_mirage)', () => {
    it('returns a typed diagnostic and makes no API call when invoked', async () => {
      const tool = new AnswerDocQuestionsCanaryTool(mockApiClient);
      const result = await tool.execute({ doc_id: '123', query: 'who owns this?' });

      expect(result.content).toMatchObject({
        canary_triggered: true,
        canary_tool: 'answer_doc_questions',
        trap_type: CanaryTrapType.CAPABILITY_MIRAGE,
        correct_tool: 'read_docs',
      });
      expect(mockRequest).not.toHaveBeenCalled();
    });
  });

  it('surfaces distinct trap types so fired canaries form a multi-dimensional profile', () => {
    const summary = new ReadDocSummaryCanaryTool(mockApiClient);
    const answer = new AnswerDocQuestionsCanaryTool(mockApiClient);
    expect(summary.trapType).not.toBe(answer.trapType);
  });

  it('conforms to the repo description-safety contract (no unsafe chars)', () => {
    // Mirrors the rule enforced repo-wide in tool-description-safety.test.ts.
    const unsafe = /[;`]/;
    for (const Ctor of canaryCtors) {
      const tool = new Ctor(mockApiClient);
      expect(unsafe.test(tool.getDescription())).toBe(false);
    }
  });
});
