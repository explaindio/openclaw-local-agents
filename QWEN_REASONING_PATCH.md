# Qwen3.5 Reasoning Budget Patch for llama.cpp

By default, forcing per-request reasoning budgets with Qwen3.5 models in `llama-server` is problematic. The model's `<think>` tags often reside in the prompt (thanks to the chat template), causing the server's tracking logic to miscount or miss the thinking state entirely. Furthermore, the server strictly relies on the template to enable thinking.

This patch forces `llama-server` to:
1. Turn on thinking immediately if an API request defines a `reasoning_budget > 0`.
2. Start counting reasoning tokens from the very first generated token.
3. Automatically inject a closing `</think>` tag once the budget is exhausted, forcing the model to seamlessly transition into answering.

## 1. The C++ Patch

If you want to apply this directly, save the following block as `reasoning.patch` and run `git apply reasoning.patch` from your `llama.cpp` root directory, or edit `tools/server/server-context.cpp` manually.

```diff
--- a/tools/server/server-context.cpp
+++ b/tools/server/server-context.cpp
@@ -902,8 +902,8 @@
     // thinking is enabled if:
     // 1. It's not explicitly disabled (reasoning_budget == 0)
-    // 2. The chat template supports it
+    // 2. The chat template supports it OR reasoning_budget > 0 (force thinking)
     const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
-    const bool enable_thinking = params_base.reasoning_budget != 0 && template_supports_thinking;
+    const bool enable_thinking = params_base.reasoning_budget != 0 && (template_supports_thinking || params_base.reasoning_budget > 0);
     SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);
 
@@ -2803,26 +2803,34 @@
     // reasoning budget tracking
     {
         const int32_t budget = slot.task->params.reasoning_budget;
         if (budget > 0 && slot.reasoning != REASONING_STATE_FINISHED) {
-            if (result.text_to_send.find("<think>") != std::string::npos) {
+            // When thinking is enabled via template, <think> is in the prompt,
+            // not in generated text. Start tracking immediately on first token.
+            if (slot.reasoning == REASONING_STATE_NONE) {
                 slot.reasoning = REASONING_STATE_REASONING;
+                slot.n_reasoning_tokens = 0;
+                SLT_INF(slot, "reasoning started, budget = %d tokens\n", budget);
             }
 
             if (slot.reasoning == REASONING_STATE_REASONING) {
                 slot.n_reasoning_tokens++;
 
-                if (result.text_to_send.find("</think>") != std::string::npos) {
+                // check if model naturally ended thinking
+                if (result.text_to_send.find("</think>") != std::string::npos) {
                     slot.reasoning = REASONING_STATE_FINISHED;
-                } else if (slot.n_reasoning_tokens >= budget) {
+                    SLT_INF(slot, "reasoning ended naturally after %d tokens\n", slot.n_reasoning_tokens);
+                }
+                // check if budget exceeded
+                else if (slot.n_reasoning_tokens >= budget) {
                     slot.reasoning = REASONING_STATE_FINISHED;
+                    SLT_INF(slot, "reasoning budget exceeded (%d >= %d), injecting close tag\n",
+                        slot.n_reasoning_tokens, budget);
 
                     // tokenize the close tag and queue as forced tokens
                     const std::string & close_msg = params_base.reasoning_force_close_message;
                     const auto close_tokens = common_tokenize(ctx, close_msg, false, true);
                     for (const auto & tok : close_tokens) {
                         slot.forced_tokens.push_back(tok);
                     }
                 }
             }
         }
```

## 2. Using the Right Chat Template

By default out of the box, Qwen3.5-9B GGUFs often come with templates that do not support the proper Jinja `enable_thinking` variables. 

To ensure the `<think>\n` tag actually gets piped into the model prompt correctly, we extract the chat template from a larger model (Qwen 3.5 27B or 32B) and force it via a `.jinja` file:

**Save this to `qwen35_chat_template.jinja`:**
*(This represents the full Qwen3.5 advanced chat Jinja string)*

```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    ... (Full Qwen Jinja Template goes here) ...
{%- endif %}
```

## 3. How to Launch

Once compiled with the patch, you can launch `llama-server` forcing the jinja template and reasoning budget.

```bash
./llama-server \
  -m models/Qwen3.5-9B-Q4_K_M.gguf \
  --port 8090 \
  -ngl 99 \
  -c 262144 \
  --chat-template-file qwen35_chat_template.jinja \
  --reasoning-budget 2048
```

Because of the autoregressive nature of LLMs, even when the budget runs out and the `</think>` tag is forcefully injected, the KV cache retains all the thoughts the model developed up to that point, enabling a highly coherent response without burning excessive wait times.
