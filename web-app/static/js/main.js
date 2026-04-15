/* ============================================================
   Webetention — main.js
   Settings modal management.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  // ── Settings Modal ─────────────────────────────────────────
  const modal       = document.getElementById('settingsModal');
  const openBtn     = document.getElementById('settingsBtn');
  const closeBtn    = document.getElementById('settingsClose');
  const saveBtn     = document.getElementById('saveSettings');
  const statusEl    = document.getElementById('settingsStatus');

  if (!modal) return;  // Not on a page with the modal

  function openModal() {
    // Load current settings from server
    fetch('/api/settings')
      .then(r => r.json())
      .then(data => {
        const apiKeyEl  = document.getElementById('s_api_key');
        const modelEl   = document.getElementById('s_model');
        const baseUrlEl = document.getElementById('s_base_url');
        if (apiKeyEl)  apiKeyEl.placeholder  = data.api_key  ? '*** (set)' : 'sk-... (leave blank for env var)';
        if (modelEl)   modelEl.value         = data.model   || '';
        if (baseUrlEl) baseUrlEl.value       = data.base_url || '';
        if (statusEl && data.source) {
          statusEl.textContent = `Active source: ${data.source}`;
          setTimeout(() => {
            if (statusEl.textContent === `Active source: ${data.source}`) {
              statusEl.textContent = '';
            }
          }, 4000);
        }
      })
      .catch(() => {});

    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    modal.classList.remove('open');
    document.body.style.overflow = '';
  }

  if (openBtn)  openBtn.addEventListener('click', openModal);
  if (closeBtn) closeBtn.addEventListener('click', closeModal);

  // Click outside modal to close
  modal.addEventListener('click', e => {
    if (e.target === modal) closeModal();
  });

  // Escape key to close
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeModal();
  });

  // Save settings
  if (saveBtn) {
    saveBtn.addEventListener('click', () => {
      const payload = {
        api_key:  document.getElementById('s_api_key')?.value  || '',
        model:    document.getElementById('s_model')?.value    || '',
        base_url: document.getElementById('s_base_url')?.value || '',
      };

      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
        .then(r => r.json())
        .then(data => {
          if (statusEl) {
            statusEl.textContent = data.message || 'Saved!';
            setTimeout(() => { statusEl.textContent = ''; }, 3000);
          }
          // Clear API key field after saving for security
          const keyField = document.getElementById('s_api_key');
          if (keyField) keyField.value = '';
        })
        .catch(() => {
          if (statusEl) statusEl.textContent = 'Error saving.';
        });
    });
  }

  // ── Auto-dismiss flash messages ────────────────────────────
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transition = 'opacity 0.4s';
      setTimeout(() => el.remove(), 400);
    }, 5000);
  });

});
