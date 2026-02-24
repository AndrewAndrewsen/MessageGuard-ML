import type { OpenClawPluginApi } from "./types.js";
declare const messageguardMlPlugin: {
    id: string;
    name: string;
    version: string;
    description: string;
    register(api: OpenClawPluginApi): void;
};
export default messageguardMlPlugin;
