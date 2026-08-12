document.querySelectorAll("[data-quiz]").forEach((quiz) => {
  const feedback = quiz.querySelector("[data-feedback]");
  quiz.querySelectorAll("button[data-correct]").forEach((button) => {
    button.addEventListener("click", () => {
      const correct = button.dataset.correct === "true";
      feedback.textContent = correct
        ? "맞았습니다. inner 10-fold는 누수 없이 target encoding을 만듭니다."
        : "다시 생각해 보세요. 실제 모델 성능은 outer 5-fold로 측정합니다.";
    });
  });
});
