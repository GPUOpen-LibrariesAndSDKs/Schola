/* No dependencies or server required. Content and notes live in index.html. */
(() => {
  "use strict";

  const slides = [...document.querySelectorAll(".slide")];
  if (!slides.length) return;

  const deck = document.getElementById("deck");
  const previous = document.getElementById("previous");
  const next = document.getElementById("next");
  const count = document.getElementById("slide-count");
  const overview = document.getElementById("overview-dialog");
  const help = document.getElementById("help-dialog");
  const notes = document.getElementById("notes-panel");
  const notesButton = document.getElementById("notes-button");
  const fullscreenButton = document.getElementById("fullscreen");
  const slideList = document.getElementById("slide-list");
  let current = 0;

  function indexFromHash() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); }
    catch { return 0; }
    const index = slides.findIndex(slide => slide.id === id);
    return index < 0 ? 0 : index;
  }

  function updateNotes() {
    const source = slides[current].querySelector(".speaker-notes");
    document.getElementById("notes-time").textContent = `Time budget: ${source?.dataset.time || "[Set time]"}`;
    document.getElementById("notes-text").textContent = source?.textContent.trim() || "No notes for this slide.";
  }

  function showSlide(index, updateHash = true) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    // Avoid leaving focus inside a slide that is about to become hidden.
    const focusedSlide = document.activeElement?.closest(".slide");
    if (focusedSlide && focusedSlide !== slides[current]) document.activeElement.blur();
    slides.forEach((slide, i) => {
      const active = i === current;
      if (!active) slide.querySelectorAll("video").forEach(video => video.pause());
      slide.classList.toggle("is-active", active);
      slide.hidden = !active;
      slide.inert = !active;
      slide.setAttribute("aria-hidden", String(!active));
    });
    count.textContent = `${String(current + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    document.getElementById("progress").style.width = `${((current + 1) / slides.length) * 100}%`;
    document.getElementById("section-name").textContent = slides[current].dataset.section || "";
    document.title = `Schola · D4 · ${slides[current].dataset.title}`;
    [...slideList.children].forEach((button, i) => {
      if (i === current) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    updateNotes();
    // Use native fragment navigation so file:// links and browser Back work.
    if (updateHash && location.hash !== `#${slides[current].id}`) location.hash = slides[current].id;
  }

  function fitDeck() {
    const scale = Math.min(deck.clientWidth / 1600, deck.clientHeight / 900);
    document.documentElement.style.setProperty("--deck-scale", String(scale));
  }

  function toggleNotes(force) {
    const open = force ?? notes.hidden;
    notes.hidden = !open;
    notesButton.setAttribute("aria-pressed", String(open));
    if (!open && notes.contains(document.activeElement)) notesButton.focus();
  }

  function openDialog(dialog) {
    toggleNotes(false);
    dialog.showModal();
    if (dialog === overview) slideList.children[current].focus();
  }

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      document.getElementById("status").textContent = "Fullscreen is unavailable. Use your browser's fullscreen command.";
    }
  }

  slides.forEach((slide, i) => {
    slide.dataset.page = `${String(i + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
    const button = document.createElement("button");
    button.type = "button";
    const number = document.createElement("span");
    number.className = "overview-number";
    number.textContent = String(i + 1).padStart(2, "0");
    const title = document.createElement("span");
    title.textContent = slide.dataset.title || slide.id;
    button.append(number, title);
    button.addEventListener("click", () => {
      overview.close();
      showSlide(i);
    });
    slideList.append(button);
  });

  previous.addEventListener("click", () => showSlide(current - 1));
  next.addEventListener("click", () => showSlide(current + 1));
  document.getElementById("overview-button").addEventListener("click", () => openDialog(overview));
  document.getElementById("help-button").addEventListener("click", () => openDialog(help));
  notesButton.addEventListener("click", () => toggleNotes());
  document.getElementById("close-notes").addEventListener("click", () => toggleNotes(false));
  fullscreenButton.addEventListener("click", toggleFullscreen);
  document.querySelectorAll("[data-close-dialog]").forEach(button => {
    button.addEventListener("click", () => button.closest("dialog").close());
  });
  document.addEventListener("fullscreenchange", () => {
    fullscreenButton.textContent = document.fullscreenElement ? "Exit fullscreen" : "Fullscreen";
    fitDeck();
  });
  if (!document.fullscreenEnabled) fullscreenButton.hidden = true;

  document.addEventListener("keydown", event => {
    // Preserve native button/link activation, dialog navigation, and text entry.
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    if (overview.open || help.open) return;
    if (event.target.closest("input, textarea, select, [contenteditable='true']")) return;
    if (event.target.closest("button, a") && [" ", "Enter"].includes(event.key)) return;
    // Space plays or pauses a focused video instead of advancing the slide.
    if (event.target.closest("video") && event.key === " ") return;
    const key = event.key.toLowerCase();
    if (["arrowright", "arrowdown", "pagedown"].includes(key) || (key === " " && !event.shiftKey)) {
      event.preventDefault();
      showSlide(current + 1);
    } else if (["arrowleft", "arrowup", "pageup"].includes(key) || (key === " " && event.shiftKey)) {
      event.preventDefault();
      showSlide(current - 1);
    } else if (key === "home" || key === "end") {
      event.preventDefault();
      showSlide(key === "home" ? 0 : slides.length - 1);
    } else if (key === "o") openDialog(overview);
    else if (key === "n") toggleNotes();
    else if (key === "f") toggleFullscreen();
    else if (key === "?") openDialog(help);
    else if (key === "escape") toggleNotes(false);
  });

  window.addEventListener("hashchange", () => showSlide(indexFromHash(), false));
  window.addEventListener("resize", fitDeck);
  document.documentElement.classList.add("js");
  showSlide(indexFromHash(), false);
  fitDeck();
})();
