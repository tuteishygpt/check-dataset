"""CSS and JavaScript for the Gradio UI."""

# CSS for dark theme and modern styling
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    min-height: 100vh;
}

.dark {
    --body-background-fill: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

.main-title {
    text-align: center;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.settings-panel {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(10px);
}

.results-panel {
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.1em !important;
    padding: 15px 30px !important;
    border-radius: 12px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4) !important;
}

.section-title {
    color: #e2e8f0;
    font-size: 1.3em;
    margin: 20px 0 15px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.verify-btn {
    padding: 8px 16px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-weight: bold;
    color: white;
    transition: transform 0.1s;
}

.verify-btn:hover {
    transform: scale(1.05);
}

.correct-btn { background: #10b981; }
.correct-btn:hover { background: #059669; }

.incorrect-btn { background: #ef4444; }
.incorrect-btn:hover { background: #dc2626; }

footer { display: none !important; }

/* Keep components rendered but out of view */
#verification_data_input, #verification_trigger_btn {
    position: fixed !important;
    top: -10000px !important;
    left: -10000px !important;
    opacity: 0 !important;
    width: 1px !important;
    height: 1px !important;
    pointer-events: none !important;
    z-index: 1 !important;
}
"""

# JS injected into <head> (reliable execution)
head_js = """
<script>
(function () {
  function gradioRoot() {
    const app = document.querySelector("gradio-app");
    if (app && app.shadowRoot) return app.shadowRoot;
    return document;
  }

  function qs(sel) {
    return gradioRoot().querySelector(sel);
  }

  function getValueEl(containerId) {
    const c = qs("#" + containerId);
    if (!c) return null;
    return c.querySelector("textarea, input");
  }

  function clickGradioButton(containerId) {
    const c = qs("#" + containerId);
    if (!c) return false;
    const btn = c.querySelector("button") || c; // IMPORTANT: click real <button> if wrapper <div> has the id
    btn.click();
    return true;
  }

  window.verifyRecord = function (id, status) {
    try {
      const el = getValueEl("verification_data_input");
      if (!el) {
        console.error("verification_data_input not found");
        return;
      }

      const payload = JSON.stringify({ id: id, status: status, ts: Date.now() });

      const proto = (el.tagName === "TEXTAREA")
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;

      const desc = Object.getOwnPropertyDescriptor(proto, "value");
      const setter = desc && desc.set;

      if (setter) setter.call(el, payload);
      else el.value = payload;

      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));

      setTimeout(() => {
        const ok = clickGradioButton("verification_trigger_btn");
        if (!ok) console.error("verification_trigger_btn not found");
      }, 0);

    } catch (e) {
      console.error("verifyRecord error:", e);
    }
  };

  // localStorage HF Token persistence
  function setupHfTokenSaver() {
    const el = getValueEl("hf_token_input");
    if (!el) return false;

    const saved = localStorage.getItem("hf_token");
    if (saved && !el.value) {
      el.value = saved;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }

    if (!el.__hfSaverAttached) {
      el.addEventListener("blur", function () {
        if (this.value && this.value.length > 5) {
          localStorage.setItem("hf_token", this.value);
        }
      });
      el.__hfSaverAttached = true;
    }
    return true;
  }

  // retry a few times because Gradio renders async
  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    const ok = setupHfTokenSaver();
    if (ok || tries >= 40) clearInterval(timer);
  }, 250);

})();
</script>
"""
