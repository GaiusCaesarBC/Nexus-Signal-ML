"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.maybeParseBetaMessage = maybeParseBetaMessage;
exports.parseBetaMessage = parseBetaMessage;
const error_1 = require("../core/error.js");
function maybeParseBetaMessage(message, params) {
    if (!params || !('parse' in (params.output_format ?? {}))) {
        return {
            ...message,
            content: message.content.map((block) => {
                if (block.type === 'text') {
                    return {
                        ...block,
                        parsed: null,
                    };
                }
                return block;
            }),
            parsed_output: null,
        };
    }
    return parseBetaMessage(message, params);
}
function parseBetaMessage(message, params) {
    let firstParsed = null;
    const content = message.content.map((block) => {
        if (block.type === 'text') {
            const parsed = parseBetaOutputFormat(params, block.text);
            if (firstParsed === null) {
                firstParsed = parsed;
            }
            return {
                ...block,
                parsed,
            };
        }
        return block;
    });
    return {
        ...message,
        content,
        parsed_output: firstParsed,
    };
}
function parseBetaOutputFormat(params, content) {
    if (params.output_format?.type !== 'json_schema') {
        return null;
    }
    try {
        if ('parse' in params.output_format) {
            return params.output_format.parse(content);
        }
        return JSON.parse(content);
    }
    catch (error) {
        throw new error_1.AnthropicError(`Failed to parse structured output: ${error}`);
    }
}
//# sourceMappingURL=beta-parser.js.map