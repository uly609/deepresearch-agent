const form = document.querySelector("#research-form");
const question = document.querySelector("#question");
const events = document.querySelector("#events");
const report = document.querySelector("#report");
const statusLabel = document.querySelector("#status");
const button = form.querySelector("button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  events.innerHTML = "";
  report.textContent = "正在研究...";
  statusLabel.textContent = "running";
  button.disabled = true;

  const response = await fetch("/api/research", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: question.value})
  });

  if (!response.ok) {
    report.textContent = await response.text();
    statusLabel.textContent = "error";
    button.disabled = false;
    return;
  }

  const {taskId} = await response.json();
  const source = new EventSource(`/api/research/${taskId}/events`);

  source.onmessage = (message) => addEvent(JSON.parse(message.data));
  ["run_started", "plan_created", "search_started", "search_finished", "sources_deduped", "sources_scored", "citations_verified"].forEach((type) => {
    source.addEventListener(type, (message) => addEvent(JSON.parse(message.data)));
  });
  source.addEventListener("run_finished", async (message) => {
    addEvent(JSON.parse(message.data));
    source.close();
    const markdown = await fetch(`/api/research/${taskId}/report`).then((res) => res.text());
    report.textContent = markdown;
    statusLabel.textContent = "completed";
    button.disabled = false;
  });
  source.onerror = () => {
    source.close();
    statusLabel.textContent = "disconnected";
    button.disabled = false;
  };
});

function addEvent(event) {
  const li = document.createElement("li");
  li.textContent = `[${event.type}] ${event.message}`;
  const time = document.createElement("time");
  time.textContent = event.createdAt;
  li.appendChild(time);
  events.appendChild(li);
  li.scrollIntoView({block: "nearest"});
}
