// ============================================================
// LangChain-Inspired Core Framework
// Provides base classes for Tools, Chains, and Tool Registry
// to orchestrate multi-step AI analysis pipelines.
//
// This is modeled after LangChain's Tool/Chain paradigm but
// built for browser extension context (no Node.js dependencies).
// ============================================================

/**
 * ToolResult — Standardized result wrapper for all tool executions.
 * Every tool returns a ToolResult so the pipeline can track
 * success/failure, timing, and data flow between steps.
 */
export class ToolResult {
    /**
     * @param {Object} options
     * @param {boolean} options.success - Whether the tool executed successfully
     * @param {*} options.data - The output data from the tool
     * @param {string} [options.error] - Error message if failed
     * @param {Object} [options.metadata] - Additional metadata about execution
     */
    constructor({ success, data, error = null, metadata = {} }) {
        this.success = success;
        this.data = data;
        this.error = error;
        this.metadata = {
            ...metadata,
            timestamp: Date.now(),
        };
    }

    /**
     * Create a successful result
     * @param {*} data - The output data
     * @param {Object} [metadata] - Optional metadata
     * @returns {ToolResult}
     */
    static ok(data, metadata = {}) {
        return new ToolResult({ success: true, data, metadata });
    }

    /**
     * Create a failed result
     * @param {string} error - Error message
     * @param {Object} [metadata] - Optional metadata
     * @returns {ToolResult}
     */
    static fail(error, metadata = {}) {
        return new ToolResult({ success: false, data: null, error, metadata });
    }

    /**
     * Check if this result has valid data
     * @returns {boolean}
     */
    hasData() {
        return this.success && this.data !== null && this.data !== undefined;
    }

    /**
     * Convert to a serializable object for message passing
     * @returns {Object}
     */
    toJSON() {
        return {
            success: this.success,
            data: this.data,
            error: this.error,
            metadata: this.metadata,
        };
    }
}

/**
 * BaseTool — Abstract base class for all tools in the pipeline.
 * Each tool has a name, description, and an execute method.
 * 
 * Inspired by LangChain's Tool class, adapted for browser context.
 * 
 * Usage:
 *   class MyTool extends BaseTool {
 *     constructor() {
 *       super({
 *         name: 'my_tool',
 *         description: 'Does something useful',
 *         version: '1.0.0',
 *       });
 *     }
 *     async _execute(input, context) {
 *       // your logic here
 *       return ToolResult.ok(processedData);
 *     }
 *   }
 */
export class BaseTool {
    /**
     * @param {Object} config
     * @param {string} config.name - Unique identifier for this tool
     * @param {string} config.description - Human-readable description
     * @param {string} [config.version] - Semantic version
     * @param {string[]} [config.requiredInputFields] - Fields required in input
     * @param {string[]} [config.outputFields] - Fields produced in output
     * @param {boolean} [config.cacheable] - Whether results can be cached
     */
    constructor({
        name,
        description,
        version = "1.0.0",
        requiredInputFields = [],
        outputFields = [],
        cacheable = false,
    }) {
        if (new.target === BaseTool) {
            throw new Error("BaseTool is abstract and cannot be instantiated directly.");
        }
        this.name = name;
        this.description = description;
        this.version = version;
        this.requiredInputFields = requiredInputFields;
        this.outputFields = outputFields;
        this.cacheable = cacheable;
        this._cache = new Map();
        this._executionCount = 0;
        this._totalExecutionTime = 0;
    }

    /**
     * Public execute method with validation, timing, and error handling.
     * Subclasses should override _execute(), not this method.
     * 
     * @param {*} input - Input data for the tool
     * @param {Object} [context] - Shared context across the pipeline
     * @returns {Promise<ToolResult>}
     */
    async execute(input, context = {}) {
        const startTime = performance.now();

        try {
            // Validate required input fields
            this._validateInput(input);

            // Check cache if applicable
            if (this.cacheable) {
                const cacheKey = this._getCacheKey(input);
                if (this._cache.has(cacheKey)) {
                    const cached = this._cache.get(cacheKey);
                    return new ToolResult({
                        ...cached,
                        metadata: {
                            ...cached.metadata,
                            fromCache: true,
                            toolName: this.name,
                        },
                    });
                }
            }

            // Execute the tool's logic
            const result = await this._execute(input, context);

            // Track execution metrics
            const executionTime = performance.now() - startTime;
            this._executionCount++;
            this._totalExecutionTime += executionTime;

            // Enrich result metadata
            result.metadata = {
                ...result.metadata,
                toolName: this.name,
                toolVersion: this.version,
                executionTimeMs: Math.round(executionTime),
                executionCount: this._executionCount,
            };

            // Cache if applicable
            if (this.cacheable && result.success) {
                const cacheKey = this._getCacheKey(input);
                this._cache.set(cacheKey, result.toJSON());
            }

            return result;
        } catch (error) {
            const executionTime = performance.now() - startTime;
            return ToolResult.fail(
                `Tool "${this.name}" failed: ${error.message}`,
                {
                    toolName: this.name,
                    executionTimeMs: Math.round(executionTime),
                    stack: error.stack,
                }
            );
        }
    }

    /**
     * Validate that required input fields are present.
     * @param {*} input
     * @throws {Error} If validation fails
     */
    _validateInput(input) {
        if (this.requiredInputFields.length === 0) return;

        if (!input || typeof input !== "object") {
            throw new Error(
                `Tool "${this.name}" requires an object input with fields: ${this.requiredInputFields.join(", ")}`
            );
        }

        for (const field of this.requiredInputFields) {
            if (!(field in input)) {
                throw new Error(
                    `Tool "${this.name}" requires input field "${field}" which is missing.`
                );
            }
        }
    }

    /**
     * Generate a cache key from the input.
     * @param {*} input
     * @returns {string}
     */
    _getCacheKey(input) {
        return JSON.stringify(input).substring(0, 500);
    }

    /**
     * Abstract method — subclasses MUST override this.
     * @param {*} input - Input data
     * @param {Object} context - Shared context
     * @returns {Promise<ToolResult>}
     */
    async _execute(input, context) {
        throw new Error(
            `Tool "${this.name}" must implement the _execute() method.`
        );
    }

    /**
     * Get execution statistics for this tool.
     * @returns {Object}
     */
    getStats() {
        return {
            name: this.name,
            version: this.version,
            executionCount: this._executionCount,
            totalExecutionTimeMs: Math.round(this._totalExecutionTime),
            averageExecutionTimeMs:
                this._executionCount > 0
                    ? Math.round(this._totalExecutionTime / this._executionCount)
                    : 0,
            cacheSize: this._cache.size,
        };
    }

    /**
     * Clear the tool's cache.
     */
    clearCache() {
        this._cache.clear();
    }

    /**
     * String representation of this tool.
     * @returns {string}
     */
    toString() {
        return `[Tool: ${this.name} v${this.version}] ${this.description}`;
    }
}

/**
 * ToolRegistry — Central registry for discovering and managing tools.
 * All tools register here so the pipeline can reference them by name.
 * 
 * Inspired by LangChain's ToolKit pattern.
 */
export class ToolRegistry {
    constructor() {
        /** @type {Map<string, BaseTool>} */
        this._tools = new Map();
        /** @type {Map<string, string[]>} Tool categories */
        this._categories = new Map();
    }

    /**
     * Register a tool instance.
     * @param {BaseTool} tool - The tool to register
     * @param {string} [category] - Optional category for grouping
     * @returns {ToolRegistry} this, for chaining
     */
    register(tool, category = "default") {
        if (!(tool instanceof BaseTool)) {
            throw new Error("Only BaseTool instances can be registered.");
        }
        if (this._tools.has(tool.name)) {
            console.warn(
                `[ToolRegistry] Overwriting existing tool: ${tool.name}`
            );
        }

        this._tools.set(tool.name, tool);

        // Track category
        if (!this._categories.has(category)) {
            this._categories.set(category, []);
        }
        this._categories.get(category).push(tool.name);

        console.log(`[ToolRegistry] Registered: ${tool.toString()}`);
        return this;
    }

    /**
     * Get a tool by name.
     * @param {string} name
     * @returns {BaseTool}
     * @throws {Error} If tool not found
     */
    get(name) {
        const tool = this._tools.get(name);
        if (!tool) {
            const available = [...this._tools.keys()].join(", ");
            throw new Error(
                `Tool "${name}" not found in registry. Available tools: ${available}`
            );
        }
        return tool;
    }

    /**
     * Check if a tool is registered.
     * @param {string} name
     * @returns {boolean}
     */
    has(name) {
        return this._tools.has(name);
    }

    /**
     * Get all tools in a category.
     * @param {string} category
     * @returns {BaseTool[]}
     */
    getByCategory(category) {
        const names = this._categories.get(category) || [];
        return names.map((name) => this._tools.get(name)).filter(Boolean);
    }

    /**
     * List all registered tools with their descriptions.
     * @returns {Object[]}
     */
    listTools() {
        return [...this._tools.values()].map((tool) => ({
            name: tool.name,
            description: tool.description,
            version: tool.version,
            outputFields: tool.outputFields,
        }));
    }

    /**
     * Get execution stats for all tools.
     * @returns {Object[]}
     */
    getAllStats() {
        return [...this._tools.values()].map((tool) => tool.getStats());
    }

    /**
     * Total number of registered tools.
     * @returns {number}
     */
    get size() {
        return this._tools.size;
    }

    /**
     * Clear all registered tools.
     */
    clear() {
        this._tools.clear();
        this._categories.clear();
    }
}

/**
 * ChainStep — A single step in a chain, wrapping a tool with
 * input/output transformation functions.
 */
export class ChainStep {
    /**
     * @param {Object} config
     * @param {string} config.toolName - Name of the tool to execute
     * @param {string} [config.label] - Human-readable label for this step
     * @param {Function} [config.inputTransform] - Transform input before tool execution
     * @param {Function} [config.outputTransform] - Transform output after tool execution
     * @param {boolean} [config.optional] - If true, chain continues on failure
     * @param {Function} [config.condition] - If provided, step only runs if condition returns true
     */
    constructor({
        toolName,
        label = null,
        inputTransform = null,
        outputTransform = null,
        optional = false,
        condition = null,
    }) {
        this.toolName = toolName;
        this.label = label || toolName;
        this.inputTransform = inputTransform;
        this.outputTransform = outputTransform;
        this.optional = optional;
        this.condition = condition;
    }
}

/**
 * Chain — Orchestrates a sequence of tools (steps) into a pipeline.
 * Each step's output feeds into the next step's input.
 * 
 * Inspired by LangChain's SequentialChain.
 * 
 * Usage:
 *   const chain = new Chain({
 *     name: 'job_analysis',
 *     registry: toolRegistry,
 *   });
 *   chain.addStep(new ChainStep({ toolName: 'link_detector' }));
 *   chain.addStep(new ChainStep({ toolName: 'link_scraper' }));
 *   chain.addStep(new ChainStep({ toolName: 'job_analyzer' }));
 *   const result = await chain.run(initialInput);
 */
export class Chain {
    /**
     * @param {Object} config
     * @param {string} config.name - Name of this chain
     * @param {ToolRegistry} config.registry - The tool registry to look up tools
     * @param {string} [config.description] - Description of what this chain does
     * @param {Function} [config.onStepStart] - Callback before each step
     * @param {Function} [config.onStepComplete] - Callback after each step
     * @param {Function} [config.onError] - Callback on step error
     */
    constructor({
        name,
        registry,
        description = "",
        onStepStart = null,
        onStepComplete = null,
        onError = null,
    }) {
        this.name = name;
        this.registry = registry;
        this.description = description;
        this.onStepStart = onStepStart;
        this.onStepComplete = onStepComplete;
        this.onError = onError;

        /** @type {ChainStep[]} */
        this.steps = [];

        /** @type {Object[]} Execution history */
        this._history = [];
    }

    /**
     * Add a step to the chain.
     * @param {ChainStep} step
     * @returns {Chain} this, for chaining
     */
    addStep(step) {
        if (!(step instanceof ChainStep)) {
            throw new Error("Steps must be ChainStep instances.");
        }
        this.steps.push(step);
        return this;
    }

    /**
     * Execute the entire chain sequentially.
     * Each step's output becomes the next step's input.
     * The context object is shared across all steps.
     * 
     * @param {*} initialInput - Initial input for the first step
     * @param {Object} [context] - Shared context object
     * @returns {Promise<ChainResult>}
     */
    async run(initialInput, context = {}) {
        const chainStartTime = performance.now();
        const stepResults = [];
        let currentInput = initialInput;

        // Initialize context with chain metadata
        context.__chain = {
            name: this.name,
            totalSteps: this.steps.length,
            startTime: Date.now(),
        };

        console.log(
            `[Chain: ${this.name}] Starting with ${this.steps.length} steps`
        );

        for (let i = 0; i < this.steps.length; i++) {
            const step = this.steps[i];
            context.__chain.currentStep = i + 1;
            context.__chain.currentStepLabel = step.label;

            // Check condition
            if (step.condition && !step.condition(currentInput, context)) {
                console.log(
                    `[Chain: ${this.name}] Skipping step ${i + 1}/${this.steps.length}: ${step.label} (condition not met)`
                );
                stepResults.push({
                    step: step.label,
                    skipped: true,
                    reason: "condition not met",
                });
                continue;
            }

            console.log(
                `[Chain: ${this.name}] Step ${i + 1}/${this.steps.length}: ${step.label}`
            );

            // Notify step start
            if (this.onStepStart) {
                this.onStepStart({
                    stepIndex: i,
                    stepLabel: step.label,
                    totalSteps: this.steps.length,
                    toolName: step.toolName,
                });
            }

            // Get the tool from registry
            const tool = this.registry.get(step.toolName);

            // Apply input transformation if provided
            let toolInput = currentInput;
            if (step.inputTransform) {
                try {
                    toolInput = step.inputTransform(currentInput, context);
                } catch (transformError) {
                    const result = ToolResult.fail(
                        `Input transform failed for step "${step.label}": ${transformError.message}`
                    );
                    stepResults.push({
                        step: step.label,
                        result: result.toJSON(),
                    });

                    if (!step.optional) {
                        return new ChainResult({
                            success: false,
                            error: result.error,
                            stepResults,
                            executionTimeMs: performance.now() - chainStartTime,
                        });
                    }
                    continue;
                }
            }

            // Execute the tool
            const result = await tool.execute(toolInput, context);

            // Apply output transformation if provided
            if (step.outputTransform && result.success) {
                try {
                    result.data = step.outputTransform(result.data, context);
                } catch (transformError) {
                    result.success = false;
                    result.error = `Output transform failed: ${transformError.message}`;
                }
            }

            stepResults.push({
                step: step.label,
                result: result.toJSON(),
            });

            // Notify step complete
            if (this.onStepComplete) {
                this.onStepComplete({
                    stepIndex: i,
                    stepLabel: step.label,
                    totalSteps: this.steps.length,
                    result: result,
                });
            }

            // Handle failure
            if (!result.success) {
                console.warn(
                    `[Chain: ${this.name}] Step "${step.label}" failed: ${result.error}`
                );

                if (this.onError) {
                    this.onError({
                        stepIndex: i,
                        stepLabel: step.label,
                        error: result.error,
                    });
                }

                if (!step.optional) {
                    return new ChainResult({
                        success: false,
                        error: `Chain failed at step "${step.label}": ${result.error}`,
                        stepResults,
                        executionTimeMs: performance.now() - chainStartTime,
                    });
                }
                // If optional, continue with previous input
                continue;
            }

            // Use the output as the next step's input
            currentInput = result.data;
        }

        const totalTime = performance.now() - chainStartTime;

        // Store in history
        this._history.push({
            timestamp: Date.now(),
            executionTimeMs: Math.round(totalTime),
            stepCount: this.steps.length,
            success: true,
        });

        console.log(
            `[Chain: ${this.name}] Completed in ${Math.round(totalTime)}ms`
        );

        return new ChainResult({
            success: true,
            data: currentInput,
            stepResults,
            executionTimeMs: totalTime,
        });
    }

    /**
     * Get execution history.
     * @returns {Object[]}
     */
    getHistory() {
        return [...this._history];
    }

    /**
     * Get a summary of this chain's steps.
     * @returns {string}
     */
    describe() {
        const lines = [`Chain: ${this.name}`];
        if (this.description) lines.push(`  ${this.description}`);
        lines.push(`  Steps:`);
        this.steps.forEach((step, i) => {
            const badge = step.optional ? "(optional)" : "(required)";
            lines.push(`    ${i + 1}. [${step.toolName}] ${step.label} ${badge}`);
        });
        return lines.join("\n");
    }
}

/**
 * ChainResult — Return value from Chain.run()
 * Contains the final output data plus metadata about each step.
 */
export class ChainResult {
    /**
     * @param {Object} options
     * @param {boolean} options.success
     * @param {*} [options.data]
     * @param {string} [options.error]
     * @param {Object[]} [options.stepResults]
     * @param {number} [options.executionTimeMs]
     */
    constructor({
        success,
        data = null,
        error = null,
        stepResults = [],
        executionTimeMs = 0,
    }) {
        this.success = success;
        this.data = data;
        this.error = error;
        this.stepResults = stepResults;
        this.executionTimeMs = Math.round(executionTimeMs);
    }

    /**
     * Get a summary of which steps succeeded/failed.
     * @returns {Object}
     */
    getSummary() {
        const completed = this.stepResults.filter(
            (s) => s.result?.success && !s.skipped
        ).length;
        const failed = this.stepResults.filter(
            (s) => s.result && !s.result.success && !s.skipped
        ).length;
        const skipped = this.stepResults.filter((s) => s.skipped).length;

        return {
            success: this.success,
            totalSteps: this.stepResults.length,
            completed,
            failed,
            skipped,
            executionTimeMs: this.executionTimeMs,
        };
    }

    /**
     * Convert to a serializable object.
     * @returns {Object}
     */
    toJSON() {
        return {
            success: this.success,
            data: this.data,
            error: this.error,
            summary: this.getSummary(),
            stepResults: this.stepResults,
        };
    }
}
