package community.deepresearch.web;

import community.deepresearch.model.ResearchEvent;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingQueue;

public final class TaskEventBus {
    private final Map<String, List<BlockingQueue<ResearchEvent>>> subscribers = new ConcurrentHashMap<>();

    public BlockingQueue<ResearchEvent> subscribe(String taskId) {
        BlockingQueue<ResearchEvent> queue = new LinkedBlockingQueue<>();
        subscribers.computeIfAbsent(taskId, ignored -> new CopyOnWriteArrayList<>()).add(queue);
        return queue;
    }

    public void unsubscribe(String taskId, BlockingQueue<ResearchEvent> queue) {
        List<BlockingQueue<ResearchEvent>> queues = subscribers.get(taskId);
        if (queues != null) {
            queues.remove(queue);
        }
    }

    public void publish(ResearchEvent event) {
        for (BlockingQueue<ResearchEvent> queue : subscribers.getOrDefault(event.taskId(), List.of())) {
            queue.offer(event);
        }
    }
}
