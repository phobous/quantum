from data.data_preprocessing import preprocess_mnist
from model.quantum_actor_critic import create_quantum_model
from model.classical_actor_critic import create_fair_classical_model
from trainer.train import train_qnn_actor_critic, train_classical_actor_critic

def main():
    # 加载数据
    x_train, y_train, x_test, y_test = preprocess_mnist()

    # 创建量子和经典神经网络模型
    quantum_actor = create_quantum_model()
    quantum_critic = create_quantum_model()

    classical_actor = create_fair_classical_model()
    classical_critic = create_fair_classical_model()

    # 训练量子模型
    print("Training Quantum Actor-Critic Model...")
    train_qnn_actor_critic(quantum_actor, quantum_critic, x_train, y_train, x_test, y_test)

    # 训练经典模型
    print("Training Classical Actor-Critic Model...")
    train_classical_actor_critic(classical_actor, classical_critic, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
