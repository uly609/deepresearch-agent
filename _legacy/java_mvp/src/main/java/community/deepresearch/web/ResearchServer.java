package community.deepresearch.web;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import community.deepresearch.agent.ResearchWorkflow;
import community.deepresearch.model.ResearchEvent;
import community.deepresearch.model.ResearchTask;
import community.deepresearch.store.RunStore;
import community.deepresearch.util.Json;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

public final class ResearchServer {
    private final HttpServer server;
    private final ResearchWorkflow workflow;
    private final RunStore runStore;
    private final ExecutorService executor;
    private final TaskEventBus eventBus = new TaskEventBus();

    public ResearchServer(int port, ResearchWorkflow workflow, RunStore runStore, ExecutorService executor) throws IOException {
        this.server = HttpServer.create(new InetSocketAddress(port), 0);
        this.workflow = workflow;
        this.runStore = runStore;
        this.executor = executor;
        server.setExecutor(executor);
        server.createContext("/", this::handle);
    }

    public void start() {
        server.start();
    }

    private void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        String method = exchange.getRequestMethod();
        try {
            if ("GET".equals(method) && "/".equals(path)) {
                staticResource(exchange, "index.html", "text/html; charset=utf-8");
            } else if ("GET".equals(method) && "/app.css".equals(path)) {
                staticResource(exchange, "app.css", "text/css; charset=utf-8");
            } else if ("GET".equals(method) && "/app.js".equals(path)) {
                staticResource(exchange, "app.js", "application/javascript; charset=utf-8");
            } else if ("POST".equals(method) && "/api/research".equals(path)) {
                createTask(exchange);
            } else if ("GET".equals(method) && path.matches("/api/research/[^/]+")) {
                getTask(exchange, path.substring("/api/research/".length()));
            } else if ("GET".equals(method) && path.matches("/api/research/[^/]+/report")) {
                String taskId = path.substring("/api/research/".length(), path.length() - "/report".length());
                report(exchange, taskId);
            } else if ("GET".equals(method) && path.matches("/api/research/[^/]+/events")) {
                String taskId = path.substring("/api/research/".length(), path.length() - "/events".length());
                events(exchange, taskId);
            } else {
                text(exchange, 404, "not found", "text/plain; charset=utf-8");
            }
        } catch (RuntimeException exception) {
            text(exchange, 500, exception.getMessage(), "text/plain; charset=utf-8");
        }
    }

    private void createTask(HttpExchange exchange) throws IOException {
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        String question = Json.readStringField(body, "question").trim();
        if (question.isEmpty()) {
            text(exchange, 400, "{\"error\":\"question is required\"}", "application/json; charset=utf-8");
            return;
        }
        ResearchTask task = new ResearchTask(question);
        runStore.put(task);
        executor.submit(() -> workflow.run(task, eventBus::publish));
        text(exchange, 202, "{\"taskId\":" + Json.quote(task.id()) + "}", "application/json; charset=utf-8");
    }

    private void getTask(HttpExchange exchange, String taskId) throws IOException {
        ResearchTask task = runStore.get(taskId);
        if (task == null) {
            text(exchange, 404, "{\"error\":\"task not found\"}", "application/json; charset=utf-8");
            return;
        }
        String json = "{\"id\":" + Json.quote(task.id())
                + ",\"question\":" + Json.quote(task.question())
                + ",\"status\":" + Json.quote(task.status())
                + ",\"sourceCount\":" + task.sources().size()
                + ",\"citationCount\":" + task.citations().size()
                + ",\"eventCount\":" + task.events().size()
                + "}";
        text(exchange, 200, json, "application/json; charset=utf-8");
    }

    private void report(HttpExchange exchange, String taskId) throws IOException {
        String report = runStore.readReport(taskId);
        if (report.isBlank()) {
            text(exchange, 404, "report not ready", "text/plain; charset=utf-8");
            return;
        }
        text(exchange, 200, report, "text/markdown; charset=utf-8");
    }

    private void events(HttpExchange exchange, String taskId) throws IOException {
        ResearchTask task = runStore.get(taskId);
        if (task == null) {
            text(exchange, 404, "task not found", "text/plain; charset=utf-8");
            return;
        }
        exchange.getResponseHeaders().add("Content-Type", "text/event-stream; charset=utf-8");
        exchange.getResponseHeaders().add("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(200, 0);
        BlockingQueue<ResearchEvent> queue = eventBus.subscribe(taskId);
        try (OutputStream output = exchange.getResponseBody()) {
            for (ResearchEvent event : task.events()) {
                writeEvent(output, event);
            }
            while (!"completed".equals(task.status())) {
                ResearchEvent event = queue.poll(30, TimeUnit.SECONDS);
                if (event == null) {
                    output.write(": keepalive\n\n".getBytes(StandardCharsets.UTF_8));
                    output.flush();
                } else {
                    writeEvent(output, event);
                }
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        } finally {
            eventBus.unsubscribe(taskId, queue);
        }
    }

    private void writeEvent(OutputStream output, ResearchEvent event) throws IOException {
        String json = "{\"type\":" + Json.quote(event.type())
                + ",\"message\":" + Json.quote(event.message())
                + ",\"createdAt\":" + Json.quote(event.createdAt().toString())
                + "}";
        output.write(("event: " + event.type() + "\n").getBytes(StandardCharsets.UTF_8));
        output.write(("data: " + json + "\n\n").getBytes(StandardCharsets.UTF_8));
        output.flush();
    }

    private void staticResource(HttpExchange exchange, String name, String contentType) throws IOException {
        try (var input = ResearchServer.class.getResourceAsStream("/static/" + name)) {
            if (input == null) {
                text(exchange, 404, "not found", "text/plain; charset=utf-8");
                return;
            }
            byte[] bytes = input.readAllBytes();
            exchange.getResponseHeaders().add("Content-Type", contentType);
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        }
    }

    private void text(HttpExchange exchange, int status, String text, String contentType) throws IOException {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", contentType);
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }
}
