import { detectSecrets, redactContent } from "./classifier.js";
const DEFAULT_MODEL_ID = "AndrewAndrewsen/distilbert-secret-masker";
const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_MASK = "[REDACTED]";
const messageguardMlPlugin = {
    id: "messageguard-ml",
    name: "MessageGuard ML",
    version: "1.0.0",
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
        api.logger.info(`MessageGuard ML: registered message_sending hook (model: ${modelId}, threshold: ${threshold})`);
        api.on("message_sending", async (event, ctx) => {
            if (!event.content)
                return;
            let spans;
            try {
                spans = await detectSecrets(event.content, { modelId, threshold });
            }
            catch (err) {
                api.logger.warn(`MessageGuard ML: model unavailable or inference failed; passing message through. ${err instanceof Error ? err.message : String(err)}`);
                return;
            }
            if (spans.length === 0)
                return;
            const redacted = redactContent(event.content, spans, mask);
            if (redacted === event.content)
                return;
            api.logger.warn(`MessageGuard ML: redacted ${spans.length} sensitive span(s) in outgoing message to ${event.to} (channel: ${ctx.channelId}).`);
            return { content: redacted };
        }, { priority: 100 });
    },
};
export default messageguardMlPlugin;
