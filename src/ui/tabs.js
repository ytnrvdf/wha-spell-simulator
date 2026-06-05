import { copyDiagnosticPanel } from "./diagnosticsView.js";

function switchRootPanel(elements, panelId) {
  const panels = [elements.dictionaryRootPanel, elements.diagnosticRootPanel, elements.shapesRootPanel];
  elements.panelTabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.panelRoot === panelId);
  });
  panels.forEach((panel) => {
    panel.hidden = panel.id !== panelId;
  });
}

export function setupTabs(elements) {
  const diagnosticPanels = [elements.parserPanelShell, elements.astPanelShell, elements.irPanelShell];
  elements.diagnosticTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      elements.diagnosticTabButtons.forEach((item) => item.classList.toggle("active", item === button));
      diagnosticPanels.forEach((panel) => {
        panel.hidden = panel.id !== `${button.dataset.diagnosticPanel}Shell`;
      });
    });
  });

  elements.diagnosticCopyButtons.forEach((button) => {
    button.addEventListener("click", () => {
      copyDiagnosticPanel(button.dataset.copyPanel, button);
    });
  });

  const dictionaryPanels = [
    elements.sampleSpellReferencePanel,
    elements.sigilReferencePanel,
    elements.signReferencePanel
  ];
  elements.dictionaryTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      elements.dictionaryTabButtons.forEach((item) => item.classList.toggle("active", item === button));
      dictionaryPanels.forEach((panel) => {
        panel.hidden = panel.id !== button.dataset.dictionaryPanel;
      });
    });
  });

  elements.panelTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      switchRootPanel(elements, button.dataset.panelRoot);
    });
  });
}
