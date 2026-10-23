import { describe, expect, it } from "vitest";
import { parseToolCallsFromText } from "../protocol/tool-parser.js";
import { extractTools } from "../protocol/transform.js";

describe("parseToolCallsFromText", () => {
  it("parses XML style <tool_call> tags with <function> and <parameter>", () => {
    const text = `The user wants to read a file.

<tool_call>
<function=read>
<parameter=file_path>
/path/to/file.md
</parameter>
</function>
</tool_call>
<tool_call>
<function=bash>
<parameter=command>
echo "hello world"
</parameter>
</function>
</tool_call>`;

    const { cleanText, toolCalls } = parseToolCallsFromText(text);
    expect(cleanText).toBe("The user wants to read a file.");
    expect(toolCalls).toHaveLength(2);

    expect(toolCalls[0].name).toBe("read");
    expect(toolCalls[0].arguments.path).toBe("/path/to/file.md");
    expect(toolCalls[0].arguments.file_path).toBe("/path/to/file.md");

    expect(toolCalls[1].name).toBe("bash");
    expect(toolCalls[1].arguments.command).toBe('echo "hello world"');
  });

  it("parses JSON style <tool_call> tags", () => {
    const text = `Calling tool:
<tool_call>
{"name": "read", "arguments": {"path": "/test/path.ts"}}
</tool_call>`;

    const { cleanText, toolCalls } = parseToolCallsFromText(text);
    expect(cleanText).toBe("Calling tool:");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].name).toBe("read");
    expect(toolCalls[0].arguments).toEqual({ path: "/test/path.ts" });
  });

  it("handles numbers and booleans in XML parameters", () => {
    const text = `<tool_call>
<function=custom_tool>
<parameter=count>42</parameter>
<parameter=enabled>true</parameter>
<parameter=config>{"key": "value"}</parameter>
</function>
</tool_call>`;

    const { cleanText, toolCalls } = parseToolCallsFromText(text);
    expect(cleanText).toBe("");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].name).toBe("custom_tool");
    expect(toolCalls[0].arguments).toEqual({
      count: 42,
      enabled: true,
      config: { key: "value" },
    });
  });

  it('parses XML style with name="..." attributes', () => {
    const text = `<tool_call>
<function name="read">
<parameter name="file_path">/test/file.md</parameter>
</function>
</tool_call>`;

    const { cleanText, toolCalls } = parseToolCallsFromText(text);
    expect(cleanText).toBe("");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].name).toBe("read");
    expect(toolCalls[0].arguments.path).toBe("/test/file.md");
  });

  it("parses markdown code blocks inside <tool_call>", () => {
    const text = `<tool_call>
\`\`\`json
{"name": "read", "arguments": {"path": "/test/file.md"}}
\`\`\`
</tool_call>`;

    const { cleanText, toolCalls } = parseToolCallsFromText(text);
    expect(cleanText).toBe("");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].name).toBe("read");
    expect(toolCalls[0].arguments).toEqual({ path: "/test/file.md" });
  });
});

describe("extractTools", () => {
  it("extracts from context.tools when present (Pi 0.85)", () => {
    const context = {
      tools: [{ name: "read", description: "Read file", parameters: {} as any }],
      messages: [],
    };
    const tools = extractTools(context);
    expect(tools).toHaveLength(1);
    expect(tools[0].name).toBe("read");
  });

  it("extracts from context.messages toolsAdded when context.tools is missing (Pi 0.86)", () => {
    const context = {
      messages: [
        {
          role: "system",
          content: "",
          toolsAdded: [
            { name: "read", description: "Read file", parameters: {} as any },
            { name: "bash", description: "Run bash", parameters: {} as any },
          ],
        } as any,
        {
          role: "system",
          content: "",
          toolsRemoved: [{ name: "read" }],
          toolsAdded: [{ name: "write", description: "Write file", parameters: {} as any }],
        } as any,
      ],
    };
    const tools = extractTools(context);
    expect(tools).toHaveLength(2);
    expect(tools.map((t) => t.name)).toEqual(["bash", "write"]);
  });
});

describe("parseToolCallsFromText argument shape", () => {
  it("degrades non-object arguments to an empty object", () => {
    // ToolCall.arguments is a JsonObject. A model that emits an array, scalar or
    // null (or a string that is not JSON) used to be passed straight through, so
    // the resulting tool call failed argument validation instead of running with
    // no arguments.
    for (const payload of [
      '{"name":"a","arguments":[1,2]}',
      '{"name":"a","arguments":null}',
      '{"name":"a","arguments":42}',
      '{"name":"a","arguments":"not json"}',
    ]) {
      const { toolCalls } = parseToolCallsFromText(`<tool_call>${payload}</tool_call>`);
      expect(toolCalls).toHaveLength(1);
      expect(toolCalls[0].arguments).toEqual({});
    }
  });

  it("keeps a well-formed object and a stringified-object argument", () => {
    const direct = parseToolCallsFromText('<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>');
    expect(direct.toolCalls[0].arguments).toEqual({ x: 1 });

    const stringified = parseToolCallsFromText('<tool_call>{"name":"a","arguments":"{\\"x\\":2}"}</tool_call>');
    expect(stringified.toolCalls[0].arguments).toEqual({ x: 2 });
  });
});
