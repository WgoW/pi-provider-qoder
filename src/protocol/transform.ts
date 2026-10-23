import type {
  AssistantMessage,
  ImageContent,
  Message,
  TextContent,
  ThinkingContent,
  Tool,
  ToolCall,
  ToolResultMessage,
} from "@earendil-works/pi-ai";
import { stripDsmlResidue } from "./thinking.js";
import { createQoderToolCallIdMapper } from "./tool-call-id.js";

/** OpenAI-style tool definition sent to the Qoder API. */
interface QoderTool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
  };
}

/** OpenAI-style tool call within an assistant message. */
interface QoderToolCall {
  id?: string;
  type: "function";
  function: { name?: string; arguments: string };
}

type QoderTextPart = { type: "text"; text: string };
type QoderImagePart = { type: "image_url"; image_url: { url: string } };
type QoderContent = string | Array<QoderTextPart | QoderImagePart>;

/** OpenAI-style message sent to the Qoder API. */
interface QoderMessage {
  role: "user" | "assistant" | "tool" | "system";
  content: QoderContent | null;
  reasoning_content?: string;
  tool_calls?: QoderToolCall[];
  tool_call_id?: string;
}

export function getContentText(msg: Message): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content
      .map((c) => {
        if (c.type === "text") return (c as TextContent).text;
        if (c.type === "thinking") return (c as ThinkingContent).thinking;
        return "";
      })
      .join("");
  }
  return "";
}

/** The image blocks of a message, in order. Empty when there are none. */
export function getContentImages(msg: Message): ImageContent[] {
  if (!Array.isArray(msg.content)) return [];
  return msg.content.filter((c): c is ImageContent => c.type === "image");
}

// fork: tools are resolved from the transcript system messages when the host
// passes a flat Context. Upstream resolves tools through pi-ai's transcript
// helpers instead.
export function extractTools(context: { tools?: Tool[]; messages?: Message[] }): Tool[] {
  if (Array.isArray(context.tools) && context.tools.length > 0) {
    return context.tools;
  }

  const tools = new Map<string, Tool>();
  if (Array.isArray(context.messages)) {
    for (const msg of context.messages) {
      if ((msg as any).role !== "system") continue;
      for (const tool of (msg as any).toolsRemoved ?? []) {
        if (tool?.name) tools.delete(tool.name);
      }
      for (const tool of (msg as any).toolsAdded ?? []) {
        if (tool?.name) tools.set(tool.name, tool);
      }
    }
  }
  return [...tools.values()];
}

export function transformTools(tools: Tool[]): QoderTool[] {
  return tools.map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    },
  }));
}

export function transformMessagesForQoder(messages: Message[]): QoderMessage[] {
  const normalizedMessages: QoderMessage[] = [];
  const mapToolCallId = createQoderToolCallIdMapper();

  // Dropping an assistant turn (below) also invalidates its tool calls: the
  // result that follows would refer to a tool_calls entry that is no longer in
  // the request, and upstreams reject that with "tool must follow a message
  // with tool_calls".
  const droppedToolCallIds = new Set<string>();

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    // Skip error or aborted messages
    if (
      msg.role === "assistant" &&
      ((msg as AssistantMessage).stopReason === "error" || (msg as AssistantMessage).stopReason === "aborted")
    ) {
      const am = msg as AssistantMessage;
      if (Array.isArray(am.content)) {
        for (const block of am.content) {
          if (block.type === "toolCall") {
            const id = (block as ToolCall).id;
            if (id) droppedToolCallIds.add(id);
          }
        }
      }
      continue;
    }

    // Drop the result too, otherwise it refers to a tool_calls entry that is
    // no longer in the request.
    if (msg.role === "toolResult" && droppedToolCallIds.has((msg as ToolResultMessage).toolCallId)) {
      continue;
    }

    if ((msg as any).role === "system") {
      let text = getContentText(msg);
      const sections = (msg as any).sections;
      if (sections && typeof sections === "object") {
        const sectionTexts = Object.values(sections).filter(
          (s): s is string => typeof s === "string" && s.trim().length > 0,
        );
        if (sectionTexts.length > 0) {
          text = text ? `${text}\n\n${sectionTexts.join("\n\n")}` : sectionTexts.join("\n\n");
        }
      }
      if (text.trim()) {
        normalizedMessages.push({
          role: "system",
          content: text,
        });
      }
    } else if (msg.role === "user") {
      let content: QoderContent = "";
      if (typeof msg.content === "string") {
        content = msg.content;
      } else if (Array.isArray(msg.content)) {
        const hasImage = msg.content.some((c) => c.type === "image");
        if (hasImage) {
          content = msg.content
            .map((c): QoderTextPart | QoderImagePart | null => {
              if (c.type === "text") {
                return { type: "text", text: (c as TextContent).text };
              }
              if (c.type === "image") {
                const img = c as ImageContent;
                return {
                  type: "image_url",
                  image_url: {
                    url: `data:${img.mimeType};base64,${img.data}`,
                  },
                };
              }
              return null;
            })
            .filter((p): p is QoderTextPart | QoderImagePart => p !== null);
        } else {
          content = getContentText(msg);
        }
      }
      normalizedMessages.push({
        role: "user",
        content,
      });
    } else if (msg.role === "assistant") {
      const am = msg as AssistantMessage;
      let content = "";
      let reasoningContent = "";
      const toolCalls: QoderToolCall[] = [];

      if (Array.isArray(am.content)) {
        for (const block of am.content) {
          if (block.type === "text") {
            content += (block as TextContent).text;
          } else if (block.type === "thinking") {
            // Replay reasoning out-of-band. Qoder returns it through
            // reasoning_content; putting it in visible content can pollute
            // later turns, and stored text may contain leaked DSML markup.
            reasoningContent += stripDsmlResidue((block as ThinkingContent).thinking);
          } else if (block.type === "toolCall") {
            const tc = block as ToolCall;
            toolCalls.push({
              id: mapToolCallId(tc.id),
              type: "function",
              function: {
                name: tc.name,
                arguments: typeof tc.arguments === "string" ? tc.arguments : JSON.stringify(tc.arguments),
              },
            });
          }
        }
      } else {
        content = am.content || "";
      }

      // Lookahead: only keep tool calls that will actually be followed by a
      // tool result. Any declared tool_call that lacks a subsequent tool response
      // triggers upstream 400 ("an assistant message with 'tool_calls' must be
      // followed by tool messages responding to each 'tool_call_id'").
      let validToolCalls = toolCalls;
      if (toolCalls.length > 0 && i < messages.length - 1) {
        const nextToolResultIds = new Set<string>();
        let hasSubsequentTurn = false;
        for (let k = i + 1; k < messages.length; k++) {
          const nextMsg = messages[k];
          if (nextMsg.role === "toolResult") {
            const trId = (nextMsg as ToolResultMessage).toolCallId;
            if (trId) nextToolResultIds.add(trId);
          } else if ((nextMsg as any).role !== "system") {
            hasSubsequentTurn = true;
            break;
          }
        }

        if (nextToolResultIds.size > 0 || hasSubsequentTurn) {
          validToolCalls = toolCalls.filter((tc) => tc.id && nextToolResultIds.has(tc.id));
          if (validToolCalls.length === 0 && !content.trim() && !reasoningContent.trim()) {
            for (const tc of toolCalls) {
              if (tc.id) droppedToolCallIds.add(tc.id);
            }
            continue;
          }
        }
      }

      // Qoder's gateway drops assistant messages whose content is null, which
      // orphans the following tool_result and makes dmodel/ultimate upstreams
      // reject the request ("tool must follow a message with tool_calls").
      // When an assistant turn has tool calls or reasoning but no visible
      // text, inject a placeholder so the gateway keeps the message and its
      // reasoning history.
      const mapped: QoderMessage = {
        role: "assistant",
        content: content || (validToolCalls.length > 0 || reasoningContent ? " " : null),
      };
      if (reasoningContent) {
        mapped.reasoning_content = reasoningContent;
      }
      if (validToolCalls.length > 0) {
        mapped.tool_calls = validToolCalls;
      }
      normalizedMessages.push(mapped);
    } else if (msg.role === "toolResult") {
      const prevMessage = normalizedMessages[normalizedMessages.length - 1];
      // Only a tool result whose id the IMMEDIATELY preceding assistant turn
      // declared may be emitted. Anything else is an orphan the upstream rejects
      // with "tool must follow a message with tool_calls":
      //  - the assistant turn was dropped above (aborted/error),
      //  - a user message sits between the call and its result,
      //  - the session resumed mid-turn and the first message is a result,
      //  - the lookahead above already trimmed that call out of tool_calls.
      // Null here means "no assistant to pair with", which is not permission to
      // pass the result through.
      const validDeclaredIds =
        prevMessage?.role === "assistant" && prevMessage.tool_calls
          ? new Set(prevMessage.tool_calls.map((tc) => tc.id).filter((id): id is string => !!id))
          : new Set<string>();

      const collectedImages: ImageContent[] = [];
      let toolCount = 0;

      let j = i;
      for (; j < messages.length && messages[j].role === "toolResult"; j++) {
        const tr = messages[j] as ToolResultMessage;
        if (droppedToolCallIds.has(tr.toolCallId)) {
          continue;
        }
        if (validDeclaredIds && !validDeclaredIds.has(tr.toolCallId)) {
          continue;
        }
        toolCount++;
        const textResult = getContentText(tr);
        const images = getContentImages(tr);
        const toolContent = textResult || (images.length > 0 ? "(see attached image)" : "(no tool output)");

        normalizedMessages.push({
          role: "tool",
          tool_call_id: mapToolCallId(tr.toolCallId),
          content: toolContent,
        });

        if (images.length > 0) {
          collectedImages.push(...images);
        }
      }

      i = j - 1;

      // A tool result may carry images — pi's `read` tool returns a text note
      // plus an `image` block for png/jpg/gif/webp/bmp, and screenshot tools do
      // the same. getContentText() maps every non-text block to "", so those
      // images were dropped silently: the TUI rendered the picture while the
      // model received only "Read image file [image/png]" and reported that it
      // could not see images.
      //
      // The OpenAI-shaped `tool` role has nowhere to put them — its content is
      // a plain string — so they follow as a separate user message, the same
      // shape the user branch above already builds. The leading label keeps the
      // model from reading a bare image as something the human just sent.
      //
      // All consecutive tool results are emitted first before any user image
      // message is appended. Interleaving a user message between tool results
      // violates the OpenAI requirement ("an assistant message with 'tool_calls'
      // must be followed by tool messages responding to each 'tool_call_id'").
      if (collectedImages.length > 0) {
        normalizedMessages.push({
          role: "user",
          content: [
            {
              type: "text",
              text: `[${collectedImages.length} image${collectedImages.length === 1 ? "" : "s"} returned by the previous tool call${toolCount > 1 ? "s" : ""}]`,
            },
            ...collectedImages.map(
              (img): QoderImagePart => ({
                type: "image_url",
                image_url: { url: `data:${img.mimeType};base64,${img.data}` },
              }),
            ),
          ],
        });
      }
    }
  }

  return normalizedMessages;
}
