
# 临界点检测与干预
class CriticalPoint:
    def __init__(self, threshold=0.02, learning_rate_factor=0.5):
        self.threshold = threshold
        self.learning_rate_factor = learning_rate_factor

    def detect(self, loss_history):
        if len(loss_history) < 2:
            return False
        # 比较最后两轮损失
        delta_loss = loss_history[-2] - loss_history[-1]
        return delta_loss < self.threshold

    def intervene(self, model, optimizer, loss_history):
        if self.detect(loss_history):
            # 当检测到临界点，自动调整学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.learning_rate_factor
            print("临界点触发，学习率已调整为", optimizer.param_groups[0]['lr'])
            return True
        return False
