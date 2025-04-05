import torch

class MultiClassClassificationMetrics:
    def __init__(self, topk=(1, 5)):
        self.topk = topk
        self.correct_topk = {k: 0 for k in topk}
        self.total = 0

    def update(self, outputs, targets):
        """
        Args:
            outputs (Tensor): Logits of shape [B, num_classes]
            targets (Tensor): Ground truth labels of shape [B]
        """
        max_k = max(self.topk)
        _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)  # [B, max_k]
        pred = pred.t()  # [max_k, B]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [max_k, B]

        batch_size = targets.size(0)
        self.total += batch_size

        for k in self.topk:
            self.correct_topk[k] += correct[:k].reshape(-1).float().sum().item()

    def compute(self):
        return {f"top_{k}_accuracy": self.correct_topk[k] / self.total for k in self.topk}

    def report(self):
        results = self.compute()
        print("\n=== Evaluation Metrics ===")
        for k in self.topk:
            print(f"Top-{k} Accuracy: {results[f'top_{k}_accuracy']:.4f}")
        print("==========================")
