export interface ParsedToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export function parseToolCallsFromText(text: string): {
  cleanText: string;
  toolCalls: ParsedToolCall[];
} {
  const toolCallRegex = /<tool_call>([\s\S]*?)<\/tool_call>/gi;
  const toolCalls: ParsedToolCall[] = [];

  let match = toolCallRegex.exec(text);
  while (match !== null) {
    let rawContent = match[1].trim();
    let name = "";
    let args: Record<string, unknown> = {};

    // Strip markdown code block fences if present (e.g. ```json ... ```)
    if (rawContent.startsWith("```")) {
      rawContent = rawContent
        .replace(/^```(?:json)?\s*/i, "")
        .replace(/\s*```$/, "")
        .trim();
    }

    if (rawContent.startsWith("{") && rawContent.endsWith("}")) {
      try {
        const parsed = JSON.parse(rawContent);
        name = parsed.name || parsed.function || parsed.tool || "";
        args = (parsed.arguments ?? parsed.parameters ?? parsed.args ?? {}) as Record<string, unknown>;
        if (typeof args === "string") {
          try {
            args = JSON.parse(args);
          } catch {}
        }
      } catch {}
    } else {
      const funcMatch = /<function(?:\s+name=|=|\s+)["']?([a-zA-Z0-9_-]+)["']?>([\s\S]*?)(?:<\/function>|$)/i.exec(
        rawContent,
      );
      if (funcMatch) {
        name = funcMatch[1];
        const paramsBlock = funcMatch[2];
        const paramRegex = /<parameter(?:\s+name=|=|\s+)["']?([a-zA-Z0-9_-]+)["']?>([\s\S]*?)<\/parameter>/gi;
        let pMatch = paramRegex.exec(paramsBlock);
        while (pMatch !== null) {
          const pName = pMatch[1];
          let pVal: unknown = pMatch[2].trim();
          const strVal = pMatch[2].trim();
          if ((strVal.startsWith("{") && strVal.endsWith("}")) || (strVal.startsWith("[") && strVal.endsWith("]"))) {
            try {
              pVal = JSON.parse(strVal);
            } catch {}
          } else if (strVal === "true") {
            pVal = true;
          } else if (strVal === "false") {
            pVal = false;
          } else if (/^-?\d+(\.\d+)?$/.test(strVal)) {
            pVal = Number(strVal);
          }
          args[pName] = pVal;
          pMatch = paramRegex.exec(paramsBlock);
        }
      }
    }

    if (name) {
      // `ToolCall.arguments` is a JsonObject; an XML/JSON payload such as
      // `{"name":"a","arguments":[1,2]}` (or a scalar/null) would violate that
      // contract and fail tool-argument validation, so a non-object value
      // degrades to no arguments instead of being passed through.
      if (args === null || typeof args !== "object" || Array.isArray(args)) args = {};
      // Normalize common parameter aliases for built-in tools
      if ((name === "read" || name === "write" || name === "edit") && !args.path) {
        if (args.file_path) args.path = args.file_path;
        else if (args.filePath) args.path = args.filePath;
      }
      if (name === "bash" && !args.command && args.cmd) {
        args.command = args.cmd;
      }
      toolCalls.push({ name, arguments: args });
    }
    match = toolCallRegex.exec(text);
  }

  const cleanText = text.replace(toolCallRegex, "").trim();
  return { cleanText, toolCalls };
}
