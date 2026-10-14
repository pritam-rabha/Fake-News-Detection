(() => {
  const input = document.getElementById("news_text");
  const counter = document.getElementById("char-count");
  const clearButton = document.getElementById("clear-input");
  const form = document.getElementById("analysis-form");
  const submitButton = document.getElementById("analyze-button");
  if (input) {
    const limit = Number(input.getAttribute("maxlength") || 12000);
    const updateCounter = () => {
      if (counter) counter.textContent = `${input.value.length.toLocaleString()} / ${limit.toLocaleString()} characters`;
    };
    input.addEventListener("input", updateCounter);
    updateCounter();
  }

  if (fileInput) fileInput.addEventListener("change", () => {
    if (fileName) fileName.textContent = fileInput.files && fileInput.files.length ? fileInput.files[0].name : "No file selected";
  });

  if (clearButton) clearButton.addEventListener("click", () => {
    if (input) input.value = "";
    if (counter && input) counter.textContent = `0 / ${Number(input.maxLength).toLocaleString()} characters`;
    if (submitButton) {
      submitButton.disabled = false;
      submitButton.removeAttribute("aria-busy");
      submitButton.style.opacity = "";
    }
    if (input) input.focus();
  });

  if (form && submitButton) form.addEventListener("submit", (event) => {
    const hasText = input && input.value.trim().length > 0;
    if (!hasText) {
      event.preventDefault();
      if (input) input.focus();
      return;
    }
    submitButton.disabled = true;
    submitButton.style.opacity = ".75";
    submitButton.setAttribute("aria-busy", "true");
  });

  const metricSelect = document.getElementById("metric-select");
  if (metricSelect) metricSelect.addEventListener("change", () => {
    const metric = metricSelect.value;
    document.querySelectorAll(".bar-group").forEach((group) => {
      group.style.opacity = group.dataset.metric === metric ? "1" : ".62";
    });
  });
})();
