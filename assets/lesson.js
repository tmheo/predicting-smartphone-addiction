document.querySelectorAll("[data-quiz]").forEach((quiz) => {
  const feedback = quiz.querySelector("[data-feedback]");
  quiz.querySelectorAll("button[data-correct]").forEach((button) => {
    button.addEventListener("click", () => {
      const correct = button.dataset.correct === "true";
      feedback.textContent = correct
        ? quiz.dataset.right || "맞았습니다."
        : quiz.dataset.wrong || "다시 생각해 보세요.";
    });
  });
});
