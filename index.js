import { detectSecrets, redactContent } from "./classifier.js";
const DEFAULT_MODEL_ID = "AndrewAndrewsen/distilbert-secret-masker";
const DEFAULT_THRESHOLD = 0.3;
const DEFAULT_MASK = "[REDACTED]";
const messageguardMlPlugin = {
    id: "messageguard-ml",
    name: "MessageGuard ML",
    version: "1.1.1",
    description: "Uses DistilBERT token classification to detect and redact secret-like content in outgoing messages.",
    register(api) {
        const cfg = (api.pluginConfig ?? {});
        if (cfg.enabled === false) {
            api.logger.info("MessageGuard ML: disabled by config, skipping hook registration.");
            return;
        }
        const modelId = cfg.modelId ?? DEFAULT_MODEL_ID;
        const threshold = typeof cfg.threshold === "number" && cfg.threshold >= 0 && cfg.threshold <= 1
            ? cfg.threshold
            : DEFAULT_THRESHOLD;
        const mask = cfg.mask ?? DEFAULT_MASK;
        // Hook 1: before_tool_call — intercept message tool sends
        // This is the primary enforcement path since message_sending doesn't fire
        // for tool sends or most channel replies in OpenClaw 2026.2.x
        api.on("before_tool_call", async (event, ctx) => {
            if (event.toolName !== "message")
                return;
            const params = event.params ?? {};
            if (params.action !== "send" && params.action !== "broadcast")
                return;
            const content = String(params.message ?? params.text ?? params.content ?? "");
            if (!content)
                return;
            let spans;
            try {
                spans = await detectSecrets(content, { modelId, threshold });
            }
            catch (err) {
                api.logger.warn(`MessageGuard ML: inference failed on tool send; passing through. ${err instanceof Error ? err.message : String(err)}`);
                return;
            }
            if (spans.length === 0)
                return;
            const redacted = redactContent(content, spans, mask);
            if (redacted === content)
                return;
            api.logger.warn(`MessageGuard ML: redacted ${spans.length} sensitive span(s) in message tool send to ${params.target} (channel: ${params.channel ?? "default"}).`);
            // Cover common param aliases so no schema variant is missed
            return { params: {
                    ...params,
                    message: redacted,
                    text: redacted,
                    content: redacted,
                    caption: redacted,
                } };
        }, { priority: 100 });
        // Hook 2: message_sending — intercept agent replies
        // Currently not fired for most outbound paths in 2026.2.x, but registered
        // so it will work when OpenClaw wires it up universally.
        api.on("message_sending", async (event, ctx) => {
            if (!event.content)
                return;
            let spans;
            try {
                spans = await detectSecrets(event.content, { modelId, threshold });
            }
            catch (err) {
                api.logger.warn(`MessageGuard ML: inference failed on reply; passing through. ${err instanceof Error ? err.message : String(err)}`);
                return;
            }
            if (spans.length === 0)
                return;
            const redacted = redactContent(event.content, spans, mask);
            if (redacted === event.content)
                return;
            api.logger.warn(`MessageGuard ML: redacted ${spans.length} sensitive span(s) in outgoing reply to ${event.to} (channel: ${ctx.channelId}).`);
            return { content: redacted };
        }, { priority: 100 });
        api.logger.info(`MessageGuard ML: registered before_tool_call + message_sending hooks (model: ${modelId}, threshold: ${threshold})`);
    },
};
export default messageguardMlPlugin;
