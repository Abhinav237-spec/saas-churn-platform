// Custom Interactive Cursor Trail Animation
document.addEventListener("DOMContentLoaded", () => {
  // Create cursor elements
  const dot = document.createElement("div");
  const outline = document.createElement("div");

  dot.classList.add("cursor-dot");
  outline.classList.add("cursor-outline");

  document.body.appendChild(dot);
  document.body.appendChild(outline);

  // Position variables
  let mouseX = 0;
  let mouseY = 0;
  let dotX = 0;
  let dotY = 0;
  let outlineX = 0;
  let outlineY = 0;

  // Tracking speed (lerp ratio)
  const dotSpeed = 1;
  const outlineSpeed = 0.12;

  // Track mouse coordinates
  window.addEventListener("mousemove", (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  // Animation loop
  function animate() {
    // Lerp dot position
    dotX += (mouseX - dotX) * dotSpeed;
    dotY += (mouseY - dotY) * dotSpeed;

    // Lerp outline position (creates the delay trail effect)
    outlineX += (mouseX - outlineX) * outlineSpeed;
    outlineY += (mouseY - outlineY) * outlineSpeed;

    // Apply styles
    dot.style.transform = `translate(${dotX}px, ${dotY}px)`;
    outline.style.transform = `translate(${outlineX - 12}px, ${outlineY - 12}px)`; // offset half of size (24px)

    requestAnimationFrame(animate);
  }
  
  // Start animation loop
  requestAnimationFrame(animate);

  // Hover animations on interactive elements
  const interactiveSelector = "a, button, select, input, textarea, .card, .tab-btn, tr, .interactive-el";
  
  function addHoverListeners() {
    const targets = document.querySelectorAll(interactiveSelector);
    targets.forEach(el => {
      // Avoid duplicate listeners
      if (el.dataset.hasCursorListener) return;
      el.dataset.hasCursorListener = "true";

      el.addEventListener("mouseenter", () => {
        outline.classList.add("cursor-hover");
      });
      el.addEventListener("mouseleave", () => {
        outline.classList.remove("cursor-hover");
      });
    });
  }

  // Initial listener bind
  addHoverListeners();

  // Watch for DOM changes to attach listeners to dynamically created elements
  const observer = new MutationObserver(addHoverListeners);
  observer.observe(document.body, { childList: true, subtree: true });
});
