export type SecretSpan = {
    start: number;
    end: number;
    score: number;
    label: string;
};
export type ClassifierConfig = {
    modelId: string;
    threshold: number;
};
export declare function detectSecrets(content: string, config: ClassifierConfig): Promise<SecretSpan[]>;
export declare function redactContent(content: string, spans: SecretSpan[], mask: string): string;
