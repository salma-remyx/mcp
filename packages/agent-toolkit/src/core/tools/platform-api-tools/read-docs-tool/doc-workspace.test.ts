import { tmpdir } from 'os';
import { join } from 'path';
import { MondayAgentToolkit } from 'src/mcp/toolkit';
import { callToolByNameAsync, callToolByNameRawAsync, createMockApiClient } from '../test-utils/mock-api-client';
import { ReadDocsTool } from './read-docs-tool';
import {
  clearWorkspace,
  exploreCachedDoc,
  extractFromCache,
  getCachedDoc,
  listCachedDocs,
  recordDocs,
} from './doc-workspace';

const TOOL_NAME = 'read_docs';
const DOC_ID = 'doc_ws_1';

// Each test gets its own isolated workspace directory so the process-global
// cache never leaks across tests or into the default tmp location.
let dirCounter = 0;
function uniqueWorkspaceDir(): string {
  dirCounter += 1;
  return join(tmpdir(), `monday-doc-ws-test-${process.pid}-${dirCounter}`);
}

describe('doc-workspace (Fetch-then-Explore persistent workspace)', () => {
  let mocks: ReturnType<typeof createMockApiClient>;
  let workspaceDir: string;

  beforeEach(() => {
    jest.clearAllMocks();
    mocks = createMockApiClient();
    jest.spyOn(MondayAgentToolkit.prototype as any, 'createApiClient').mockReturnValue(mocks.mockApiClient);
    workspaceDir = uniqueWorkspaceDir();
    process.env.MONDAY_DOC_WORKSPACE_DIR = workspaceDir;
  });

  afterEach(() => {
    clearWorkspace();
    delete process.env.MONDAY_DOC_WORKSPACE_DIR;
    jest.restoreAllMocks();
  });

  // ─── module-level unit tests ──────────────────────────────────────────────

  describe('workspace primitives', () => {
    it('records and retrieves a cached document', () => {
      recordDocs([{ id: DOC_ID, name: 'Memo', blocks_as_markdown: '# Hello' }]);
      const cached = getCachedDoc(DOC_ID);
      expect(cached).not.toBeNull();
      expect(cached?.blocks_as_markdown).toBe('# Hello');
      expect(cached?.name).toBe('Memo');
    });

    it('lists cached doc ids', () => {
      recordDocs([
        { id: 'a', blocks_as_markdown: 'A' },
        { id: 'b', blocks_as_markdown: 'B' },
      ]);
      expect(
        listCachedDocs()
          .map((d) => d.id)
          .sort(),
      ).toEqual(['a', 'b']);
    });

    it('extracts blocks by id', () => {
      recordDocs([
        {
          id: DOC_ID,
          blocks_as_markdown: 'ignored',
          blocks: [{ id: 'b1', type: 'text', content: JSON.stringify({ text: 'alpha' }) }],
        },
      ]);
      const result = extractFromCache(DOC_ID, { block_ids: ['b1'] });
      expect('found' in result).toBe(false);
      if ('found' in result) return;
      expect(result.matches).toHaveLength(1);
      expect(result.matches[0].block_id).toBe('b1');
      expect(result.matches[0].markdown).toBe('alpha');
    });

    it('extracts markdown lines by keyword (case-insensitive)', () => {
      recordDocs([{ id: DOC_ID, blocks_as_markdown: '# Plan\n\nBudget is 1000\n\nTimeline is Q3' }]);
      const result = extractFromCache(DOC_ID, { query: 'budget' });
      if ('found' in result) throw new Error('expected a hit');
      expect(result.total_matches).toBe(1);
      expect(result.matches[0].markdown).toBe('Budget is 1000');
    });

    it('returns the whole cached doc when no filter is given', () => {
      recordDocs([{ id: DOC_ID, blocks_as_markdown: 'full body' }]);
      const result = extractFromCache(DOC_ID, {});
      if ('found' in result) throw new Error('expected a hit');
      expect(result.total_matches).toBe(1);
      expect(result.matches[0].markdown).toBe('full body');
    });

    it('reports not-found for an unknown doc', () => {
      expect(extractFromCache('missing')).toEqual({ found: false });
      const explored = exploreCachedDoc('missing');
      expect(explored.content).toEqual(expect.stringContaining('not in the workspace cache'));
    });

    it('clearWorkspace removes recorded documents', () => {
      recordDocs([{ id: DOC_ID, blocks_as_markdown: 'x' }]);
      expect(getCachedDoc(DOC_ID)).not.toBeNull();
      clearWorkspace();
      expect(getCachedDoc(DOC_ID)).toBeNull();
    });
  });

  // ─── integration with the existing read_docs tool (the wiring) ────────────

  describe('read_docs integration', () => {
    const mockDoc = {
      id: DOC_ID,
      object_id: 'obj_1',
      name: 'Plan Doc',
      doc_kind: 'private',
      created_at: '2026-03-01T00:00:00Z',
      created_by: { name: 'Alice' },
      url: 'https://monday.com/doc/1',
      relative_url: '/doc/1',
      workspace: { name: 'WS' },
      workspace_id: 'ws_1',
      doc_folder_id: null,
      settings: null,
    };

    const markdown = '# Project Plan\n\nBudget is 1000\n\nTimeline is Q3';

    it('records a fetched doc so explore_cache can re-extract WITHOUT another API call', async () => {
      mocks.mockRequest.mockResolvedValueOnce({ docs: [mockDoc] }).mockResolvedValueOnce({
        export_markdown_from_doc: { success: true, markdown },
      });

      // Fetch (selection): a real content-mode read records the doc to the workspace.
      const fetched = await callToolByNameAsync(TOOL_NAME, { type: 'ids', ids: [DOC_ID] });
      expect(fetched.data).toHaveLength(1);
      const apiCallsAfterFetch = mocks.getMockRequest().mock.calls.length;
      expect(apiCallsAfterFetch).toBe(2); // readDocs + exportMarkdown

      // The doc is now persisted on the filesystem, independent of context.
      const cached = getCachedDoc(DOC_ID);
      expect(cached?.blocks_as_markdown).toBe(markdown);

      // Explore (extraction): pull just the budget line, no API round-trip.
      const explored = await callToolByNameAsync(TOOL_NAME, { mode: 'explore_cache', ids: [DOC_ID], query: 'budget' });

      expect(explored.doc_id).toBe(DOC_ID);
      expect(explored.total_matches).toBe(1);
      expect(explored.matches[0].markdown).toBe('Budget is 1000');

      // Decoupling assertion: extraction made zero additional API calls.
      expect(mocks.getMockRequest().mock.calls.length).toBe(apiCallsAfterFetch);
    });

    it('explore_cache reports not-cached with a helpful hint when the doc was never fetched', async () => {
      const result = await callToolByNameRawAsync(TOOL_NAME, { mode: 'explore_cache', ids: ['never_fetched'] });
      expect(result.content[0].text).toContain('not in the workspace cache');
      expect(result.content[0].text).toContain('content');
      expect(mocks.getMockRequest()).not.toHaveBeenCalled();
    });

    it('extracts specific blocks when include_blocks was true on the prior fetch', async () => {
      const docWithBlocks = {
        ...mockDoc,
        blocks: [
          {
            id: 'block_a',
            type: 'normal_text',
            parent_block_id: null,
            position: '0',
            content: JSON.stringify({ text: 'Alpha block' }),
          },
          {
            id: 'block_b',
            type: 'normal_text',
            parent_block_id: null,
            position: '1',
            content: JSON.stringify({ text: 'Beta block' }),
          },
        ],
      };
      mocks.mockRequest.mockResolvedValueOnce({ docs: [docWithBlocks] }).mockResolvedValueOnce({
        export_markdown_from_doc: { success: true, markdown },
      });

      await callToolByNameAsync(TOOL_NAME, { type: 'ids', ids: [DOC_ID], include_blocks: true });

      const explored = await callToolByNameAsync(TOOL_NAME, {
        mode: 'explore_cache',
        ids: [DOC_ID],
        block_ids: ['block_b'],
      });

      expect(explored.total_matches).toBe(1);
      expect(explored.matches[0].block_id).toBe('block_b');
      expect(explored.matches[0].markdown).toBe('Beta block');
    });

    it('exposes explore_cache mode in the tool schema and description', () => {
      const tool = new ReadDocsTool(mocks.mockApiClient);
      expect(tool.getInputSchema().mode).toBeDefined();
      expect(tool.getInputSchema().query).toBeDefined();
      expect(tool.getInputSchema().block_ids).toBeDefined();
      expect(tool.getDescription()).toContain('explore_cache');
    });
  });
});
