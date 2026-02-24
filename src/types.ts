/**
 * Minimal type stubs for the OpenClaw plugin API.
 * Full types are available from `openclaw/plugin-sdk` in the openclaw package.
 */

export type PluginLogger = {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

export type MessageSendingEvent = {
  to: string;
  content: string;
  metadata?: Record<string, unknown>;
};

export type MessageContext = {
  channelId: string;
  accountId?: string;
  conversationId?: string;
};

export type MessageSendingResult = {
  content?: string;
  cancel?: boolean;
};

export type OpenClawPluginApi = {
  id: string;
  name: string;
  pluginConfig?: Record<string, unknown>;
  logger: PluginLogger;
  resolvePath: (input: string) => string;
  on: (
    hookName: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    handler: (event: any, ctx: any) => any,
    opts?: { priority?: number }
  ) => void;
};
