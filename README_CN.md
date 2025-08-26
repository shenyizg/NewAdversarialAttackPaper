# Latest Adversarial Attack Papers
**update at 2025-08-26 10:53:46**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

引导对话动力学，增强抵御多回合越狱攻击的稳健性 cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .

摘要: 事实证明，大型语言模型（LLM）很容易受到越狱攻击，其中对抗性提示旨在引发有害反应。虽然现有的防御措施通过检测和过滤不安全的输入有效地减轻了单回合攻击，但它们无法对抗利用多次交互中的上下文漂移的多回合越狱，从而逐渐导致LLM远离安全行为。为了应对这一挑战，我们提出了一个基于安全控制理论的安全引导框架，确保多回合对话中不变的安全性。我们的方法使用状态空间表示对与LLM的对话进行建模，并引入一种新型的神经屏障函数（NBF）来主动检测和过滤不断变化的上下文中出现的有害查询。我们的方法通过学习一个考虑对抗性查询的安全预测器，在每一轮对话中实现不变的安全性，防止潜在的上下文漂移到越狱。在多个LLM下进行的大量实验表明，我们基于NBF的安全转向优于安全对准，基于转向的转向和轻型LLM护栏基线，为多转向越狱提供更强的防御，同时在安全性，有用性和过度拒绝之间保持更好的权衡。查看网站https://sites.google.com/view/llm-nbf/home。我们的代码可以在https://github.com/HanjiangHu/NBF-LLM上找到。



## **2. Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks**

传输样式以减少纹理偏差并提高语义分割网络中的鲁棒性 cs.CV

accepted at ECAI 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.10239v2) [paper-pdf](http://arxiv.org/pdf/2507.10239v2)

**Authors**: Ben Hamscher, Edgar Heinert, Annika Mütze, Kira Maag, Matthias Rottmann

**Abstract**: Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.

摘要: 最近的研究调查了深度神经网络（DNN）在图像分类中的形状和纹理偏差，这些偏差会影响其泛化能力和鲁棒性。研究表明，与常规DNN训练相比，使用风格化图像进行训练可以减少图像分类中的纹理偏差，并提高图像损坏的鲁棒性。为了推进这一领域的研究，我们研究了风格转移是否同样可以在语义分割中产生这两种效果。为此，我们执行风格转移，风格在人工图像区域之间有所不同。这些随机区域由选定数量的Voronoi细胞形成。然后使用生成的风格传输数据来训练语义分割DNN，目标是减少它们对纹理线索的依赖，同时增强它们对基于形状的特征的依赖。在我们的实验中，事实证明，在语义分割中，风格转移增强减少了纹理偏差，并大大提高了针对常见图像损坏和对抗攻击的鲁棒性。这些观察结果适用于Cityscapes数据集和Pascal Content上的卷积神经网络和Transformer架构，显示了所提出方法的通用性。



## **3. Quantum-Classical Hybrid Framework for Zero-Day Time-Push GNSS Spoofing Detection**

用于零日时间推送式全球导航卫星欺骗检测的量子经典混合框架 cs.LG

This work has been submitted to the IEEE Internet of Things Journal  for possible publication

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18085v1) [paper-pdf](http://arxiv.org/pdf/2508.18085v1)

**Authors**: Abyad Enan, Mashrur Chowdhury, Sagar Dasgupta, Mizanur Rahman

**Abstract**: Global Navigation Satellite Systems (GNSS) are critical for Positioning, Navigation, and Timing (PNT) applications. However, GNSS are highly vulnerable to spoofing attacks, where adversaries transmit counterfeit signals to mislead receivers. Such attacks can lead to severe consequences, including misdirected navigation, compromised data integrity, and operational disruptions. Most existing spoofing detection methods depend on supervised learning techniques and struggle to detect novel, evolved, and unseen attacks. To overcome this limitation, we develop a zero-day spoofing detection method using a Hybrid Quantum-Classical Autoencoder (HQC-AE), trained solely on authentic GNSS signals without exposure to spoofed data. By leveraging features extracted during the tracking stage, our method enables proactive detection before PNT solutions are computed. We focus on spoofing detection in static GNSS receivers, which are particularly susceptible to time-push spoofing attacks, where attackers manipulate timing information to induce incorrect time computations at the receiver. We evaluate our model against different unseen time-push spoofing attack scenarios: simplistic, intermediate, and sophisticated. Our analysis demonstrates that the HQC-AE consistently outperforms its classical counterpart, traditional supervised learning-based models, and existing unsupervised learning-based methods in detecting zero-day, unseen GNSS time-push spoofing attacks, achieving an average detection accuracy of 97.71% with an average false negative rate of 0.62% (when an attack occurs but is not detected). For sophisticated spoofing attacks, the HQC-AE attains an accuracy of 98.23% with a false negative rate of 1.85%. These findings highlight the effectiveness of our method in proactively detecting zero-day GNSS time-push spoofing attacks across various stationary GNSS receiver platforms.

摘要: 全球导航卫星系统（GNSS）对于定位、导航和授时（PNT）应用至关重要。然而，全球导航卫星系统非常容易受到欺骗攻击，对手会发送伪造信号来误导接收器。此类攻击可能会导致严重的后果，包括导航错误、数据完整性受损和运营中断。大多数现有的欺骗检测方法依赖于监督学习技术，并且很难检测新颖的、进化的和不可见的攻击。为了克服这一限制，我们使用混合量子经典自动编码器（HQC-AE）开发了一种零日欺骗检测方法，该方法仅根据真实的GPS信号进行训练，而不会暴露于欺骗数据。通过利用在跟踪阶段提取的特征，我们的方法能够在计算PNT解决方案之前进行主动检测。我们重点关注静态GNSS接收器中的欺骗检测，这些接收器特别容易受到时间推送欺骗攻击，攻击者操纵计时信息以在接收器上引发不正确的时间计算。我们针对不同不可见的时间推送欺骗攻击场景来评估我们的模型：简单化、中间化和复杂化。我们的分析表明，HQC-AE在检测零日、不可见的GNSS时间推送欺骗攻击方面始终优于其经典对应物、传统的基于监督学习的模型和现有的基于无监督学习的方法，实现了97.71%的平均检测准确率，平均误报率为0.62%（当攻击发生但未被检测到时）。对于复杂的欺骗攻击，HQC-AE的准确率为98.23%，误报率为1.85%。这些发现凸显了我们的方法在跨各种固定的GNSS接收器平台主动检测零日GNSS时间推送欺骗攻击方面的有效性。



## **4. Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering**

通过基于损失的客户端集群实现对抗性攻击下的鲁棒联邦学习 cs.LG

16 pages, 5 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.12672v3) [paper-pdf](http://arxiv.org/pdf/2508.12672v3)

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework.

摘要: 联合学习（FL）支持跨多个客户端的协作模型训练，而无需共享私有数据。我们考虑FL场景，其中FL客户端受到对抗性（拜占庭）攻击，而FL服务器是可信的（诚实的），并有一个值得信赖的侧数据集。这可以对应于，例如，服务器在联合之前拥有受信任数据，或者存在暂时承担服务器角色的受信任客户端的情况。我们的方法只需要两个诚实的参与者，即服务器和一个客户端，在不了解恶意客户端数量的情况下有效运行。理论分析表明，即使在强大的拜占庭攻击下，也存在有限的最优性差距。实验结果表明，我们的算法显着优于标准和强大的FL基线，如平均值，修剪平均值，中位数，克鲁姆，和多克鲁姆下的各种攻击策略，包括标签翻转，符号翻转，高斯噪声添加在MNIST，FMNIST，和CIFAR-10基准使用花框架。



## **5. FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning**

FedGreed：一种用于联邦学习的拜占庭鲁棒的基于损失的聚合方法 cs.LG

8 pages, 4 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18060v1) [paper-pdf](http://arxiv.org/pdf/2508.18060v1)

**Authors**: Emmanouil Kritharakis, Antonios Makris, Dusan Jakovetic, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients while preserving data privacy by keeping local datasets on-device. In this work, we address FL settings where clients may behave adversarially, exhibiting Byzantine attacks, while the central server is trusted and equipped with a reference dataset. We propose FedGreed, a resilient aggregation strategy for federated learning that does not require any assumptions about the fraction of adversarial participants. FedGreed orders clients' local model updates based on their loss metrics evaluated against a trusted dataset on the server and greedily selects a subset of clients whose models exhibit the minimal evaluation loss. Unlike many existing approaches, our method is designed to operate reliably under heterogeneous (non-IID) data distributions, which are prevalent in real-world deployments. FedGreed exhibits convergence guarantees and bounded optimality gaps under strong adversarial behavior. Experimental evaluations on MNIST, FMNIST, and CIFAR-10 demonstrate that our method significantly outperforms standard and robust federated learning baselines, such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum, in the majority of adversarial scenarios considered, including label flipping and Gaussian noise injection attacks. All experiments were conducted using the Flower federated learning framework.

摘要: 联合学习（FL）支持跨多个客户端进行协作模型训练，同时通过将本地数据集保留在设备上来保护数据隐私。在这项工作中，我们解决了FL设置，其中客户端可能表现出敌对行为，表现出拜占庭式攻击，而中央服务器是受信任的并配备了参考数据集。我们提出FedGreed，这是一种针对联邦学习的弹性聚合策略，不需要对对抗参与者的比例进行任何假设。FedGreed根据针对服务器上受信任数据集评估的损失指标来订购客户的本地模型更新，并贪婪地选择其模型表现出最小评估损失的客户子集。与许多现有方法不同，我们的方法旨在在现实世界部署中普遍存在的异类（非IID）数据分布下可靠运行。FedGreed在强对抗行为下表现出收敛保证和有界最优性差距。对MNIST、FMNIST和CIFAR-10的实验评估表明，在所考虑的大多数对抗场景中，我们的方法显着优于标准和稳健的联邦学习基线，例如Mean、Trimmed Mean、Median、Krum和Multi-Krum。所有实验都使用Flower联邦学习框架进行。



## **6. Does simple trump complex? Comparing strategies for adversarial robustness in DNNs**

简单胜过复杂吗？比较DNN中对抗鲁棒性的策略 cs.LG

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18019v1) [paper-pdf](http://arxiv.org/pdf/2508.18019v1)

**Authors**: William Brooks, Marelie H. Davel, Coenraad Mouton

**Abstract**: Deep Neural Networks (DNNs) have shown substantial success in various applications but remain vulnerable to adversarial attacks. This study aims to identify and isolate the components of two different adversarial training techniques that contribute most to increased adversarial robustness, particularly through the lens of margins in the input space -- the minimal distance between data points and decision boundaries. Specifically, we compare two methods that maximize margins: a simple approach which modifies the loss function to increase an approximation of the margin, and a more complex state-of-the-art method (Dynamics-Aware Robust Training) which builds upon this approach. Using a VGG-16 model as our base, we systematically isolate and evaluate individual components from these methods to determine their relative impact on adversarial robustness. We assess the effect of each component on the model's performance under various adversarial attacks, including AutoAttack and Projected Gradient Descent (PGD). Our analysis on the CIFAR-10 dataset reveals which elements most effectively enhance adversarial robustness, providing insights for designing more robust DNNs.

摘要: 深度神经网络（DNN）在各种应用中取得了巨大成功，但仍然容易受到对抗攻击。这项研究旨在识别和隔离两种不同对抗训练技术中对提高对抗鲁棒性贡献最大的组成部分，特别是通过输入空间中的裕度（数据点和决策边界之间的最小距离）的视角。具体来说，我们比较了两种最大限度地提高利润率的方法：一种修改损失函数以增加利润率逼近的简单方法，以及一种基于这种方法的更复杂的最先进方法（动态感知稳健训练）。使用VGG-16模型作为基础，我们系统地分离和评估这些方法中的各个成分，以确定它们对对抗稳健性的相对影响。我们评估了每个组件在各种对抗攻击（包括AutoAttack和投影梯度下降（PVD））下对模型性能的影响。我们对CIFAR-10数据集的分析揭示了哪些元素最有效地增强了对抗稳健性，为设计更稳健的DNN提供了见解。



## **7. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

两阶段学习推迟中的对抗稳健性：算法和保证 stat.ML

Accepted at the 42nd International Conference on Machine Learning  (ICML 2025)

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2502.01027v4) [paper-pdf](http://arxiv.org/pdf/2502.01027v4)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Two-stage Learning-to-Defer (L2D) enables optimal task delegation by assigning each input to either a fixed main model or one of several offline experts, supporting reliable decision-making in complex, multi-agent environments. However, existing L2D frameworks assume clean inputs and are vulnerable to adversarial perturbations that can manipulate query allocation--causing costly misrouting or expert overload. We present the first comprehensive study of adversarial robustness in two-stage L2D systems. We introduce two novel attack strategie--untargeted and targeted--which respectively disrupt optimal allocations or force queries to specific agents. To defend against such threats, we propose SARD, a convex learning algorithm built on a family of surrogate losses that are provably Bayes-consistent and $(\mathcal{R}, \mathcal{G})$-consistent. These guarantees hold across classification, regression, and multi-task settings. Empirical results demonstrate that SARD significantly improves robustness under adversarial attacks while maintaining strong clean performance, marking a critical step toward secure and trustworthy L2D deployment.

摘要: 两阶段学习延迟（L2 D）通过将每个输入分配给固定的主模型或多个离线专家之一来实现最佳任务委托，支持复杂的多代理环境中的可靠决策。然而，现有的L2 D框架假设干净的输入，并且容易受到可以操纵查询分配的对抗性扰动的影响，从而导致代价高昂的错误路由或专家超载。我们首次对两阶段L2 D系统中的对抗鲁棒性进行了全面研究。我们引入了两种新颖的攻击策略--无针对性和有针对性--它们分别扰乱最佳分配或强制向特定代理进行查询。为了抵御此类威胁，我们提出了SAARD，这是一种凸学习算法，它建立在一系列可证明Bayes-一致且$（\mathCal{R}，\mathCal{G}）$-一致的替代损失之上。这些保证适用于分类、回归和多任务设置。经验结果表明，SAARD显着提高了对抗攻击下的鲁棒性，同时保持了强大的清洁性能，标志着迈向安全且值得信赖的L2 D部署的关键一步。



## **8. A Predictive Framework for Adversarial Energy Depletion in Inbound Threat Scenarios**

远程威胁场景中对抗性能源消耗的预测框架 eess.SY

7 pages, 1 figure, 1 table, preprint submitted to the American  Control Conference (ACC) 2026

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17805v1) [paper-pdf](http://arxiv.org/pdf/2508.17805v1)

**Authors**: Tam W. Nguyen

**Abstract**: This paper presents a predictive framework for adversarial energy-depletion defense against a maneuverable inbound threat (IT). The IT solves a receding-horizon problem to minimize its own energy while reaching a high-value asset (HVA) and avoiding interceptors and static lethal zones modeled by Gaussian barriers. Expendable interceptors (EIs), coordinated by a central node (CN), maintain proximity to the HVA and patrol centers via radius-based tether costs, deny attack corridors by harassing and containing the IT, and commit to intercept only when a geometric feasibility test is confirmed. No explicit opponent-energy term is used, and the formulation is optimization-implementable. No simulations are included.

摘要: 本文提出了一个针对可攻击的入境威胁（IT）的对抗性能量耗尽防御的预测框架。IT解决了后退问题，以最大限度地减少自己的能量，同时获得高价值资产（HVA）并避免拦截器和由高斯屏障建模的静态致命区。由中心节点（CN）协调的消耗性拦截器（EI）通过基于半径的系绳成本保持与HVA和巡逻中心的接近性，通过骚扰和遏制IT来拒绝攻击走廊，并仅在几何可行性测试得到确认时承诺拦截。没有使用明确的余能项，并且该公式是可优化实现的。不包括模拟。



## **9. Robustness Feature Adapter for Efficient Adversarial Training**

用于高效对抗训练的鲁棒性特征适配器 cs.LG

The paper has been accepted for presentation at ECAI 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17680v1) [paper-pdf](http://arxiv.org/pdf/2508.17680v1)

**Authors**: Quanwei Wu, Jun Guo, Wei Wang, Yi Wang

**Abstract**: Adversarial training (AT) with projected gradient descent is the most popular method to improve model robustness under adversarial attacks. However, computational overheads become prohibitively large when AT is applied to large backbone models. AT is also known to have the issue of robust overfitting. This paper contributes to solving both problems simultaneously towards building more trustworthy foundation models. In particular, we propose a new adapter-based approach for efficient AT directly in the feature space. We show that the proposed adapter-based approach can improve the inner-loop convergence quality by eliminating robust overfitting. As a result, it significantly increases computational efficiency and improves model accuracy by generalizing adversarial robustness to unseen attacks. We demonstrate the effectiveness of the new adapter-based approach in different backbone architectures and in AT at scale.

摘要: 采用投影梯度下降的对抗训练（AT）是提高模型在对抗攻击下鲁棒性的最常用方法。然而，当AT应用于大型骨干模型时，计算开销变得过大。AT还已知具有鲁棒过拟合的问题。本文有助于同时解决这两个问题，建立更值得信赖的基础模型。特别是，我们提出了一个新的适配器为基础的方法，直接在特征空间中的高效AT。我们表明，所提出的基于自适应的方法可以通过消除鲁棒过拟合来提高内环收敛质量。因此，它通过将对抗性鲁棒性推广到不可见的攻击来显着提高计算效率并提高模型准确性。我们证明了新的适配器为基础的方法在不同的骨干架构和AT的规模的有效性。



## **10. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17674v1) [paper-pdf](http://arxiv.org/pdf/2508.17674v1)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **11. TombRaider: Entering the Vault of History to Jailbreak Large Language Models**

TombRaider：进入历史宝库越狱大型语言模型 cs.CR

Main Conference of EMNLP

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2501.18628v2) [paper-pdf](http://arxiv.org/pdf/2501.18628v2)

**Authors**: Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li

**Abstract**: Warning: This paper contains content that may involve potentially harmful behaviours, discussed strictly for research purposes.   Jailbreak attacks can hinder the safety of Large Language Model (LLM) applications, especially chatbots. Studying jailbreak techniques is an important AI red teaming task for improving the safety of these applications. In this paper, we introduce TombRaider, a novel jailbreak technique that exploits the ability to store, retrieve, and use historical knowledge of LLMs. TombRaider employs two agents, the inspector agent to extract relevant historical information and the attacker agent to generate adversarial prompts, enabling effective bypassing of safety filters. We intensively evaluated TombRaider on six popular models. Experimental results showed that TombRaider could outperform state-of-the-art jailbreak techniques, achieving nearly 100% attack success rates (ASRs) on bare models and maintaining over 55.4% ASR against defence mechanisms. Our findings highlight critical vulnerabilities in existing LLM safeguards, underscoring the need for more robust safety defences.

摘要: 警告：本文包含可能涉及潜在有害行为的内容，严格出于研究目的进行讨论。   越狱攻击可能会阻碍大型语言模型（LLM）应用程序的安全性，尤其是聊天机器人。研究越狱技术是提高这些应用安全性的一项重要人工智能红色团队任务。本文中，我们介绍了TombRaider，这是一种新型越狱技术，它利用了存储、检索和使用LLM历史知识的能力。TombRaider使用两个代理，检查员代理提取相关历史信息，攻击者代理生成对抗提示，从而有效绕过安全过滤器。我们对TombRaider的六款热门型号进行了深入评估。实验结果表明，TombRaider的性能优于最先进的越狱技术，在裸模型上实现了近100%的攻击成功率（ASB），并在防御机制下保持超过55.4%的ASB。我们的调查结果强调了现有LLM保障措施中的关键漏洞，强调了更强大的安全防御的必要性。



## **12. Defending against Jailbreak through Early Exit Generation of Large Language Models**

通过早期退出生成大型语言模型抵御越狱 cs.AI

ICONIP 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2408.11308v2) [paper-pdf](http://arxiv.org/pdf/2408.11308v2)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. We introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85% in comparison with 50% for the present SOTAs, with minimal impact on the utility of LLMs.

摘要: 大型语言模型（LLM）在各种应用中越来越受到关注。尽管如此，随着一些用户试图利用这些模型进行恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了降低此类风险，“对齐”技术的概念被开发出来。然而，最近的研究表明，使用复杂的即时工程或对抗性后缀（一种被称为“越狱”的技术）可能会破坏这种对齐。“我们的研究从LLM的类人类生成过程中汲取线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示的嵌入。利用这一发现，我们建议利用LLM的早期Transformer输出作为检测恶意输入并立即终止生成的手段。我们为LLM引入了一种简单但重要的防御方法，称为EEG-Defender。我们对三种模型的十种越狱方法进行了全面实验。我们的结果表明，EEG-Defender能够大幅降低攻击成功率（ASB），大约为85%，而当前SOTA的攻击成功率为50%，对LLM的实用性影响最小。



## **13. SoK: Cybersecurity Assessment of Humanoid Ecosystem**

SoK：类人生物生态系统的网络安全评估 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17481v1) [paper-pdf](http://arxiv.org/pdf/2508.17481v1)

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.

摘要: 类人猿正在向医疗保健、工业、国防和服务业的实际部署迈进。虽然通常被认为是网络物理系统（CPS），但它们对传统网络软件栈的依赖（例如，Linux操作系统）、机器人操作系统（LOS）中间件和空中更新通道创建了一个独特的安全配置文件，使它们暴露在传统CPS模型无法完全解决的漏洞中。之前的研究主要检查了特定的威胁，例如LiDART欺骗或对抗性机器学习（ML）。这种狭隘的焦点忽视了针对一个组件的攻击如何对整个机器人的互连系统造成连锁伤害。我们通过知识系统化（SoK）来解决这一差距，该知识采用全面的方法，整合机器人、CPS和网络安全领域的碎片化研究。我们为人形机器人引入了一个七层安全模型，组织了整个人形生态系统中的39种已知攻击和35种防御--从硬件到人机交互。在此安全模型的基础上，我们开发了一个具有风险加权评分的量化39 x35攻击-防御矩阵，并通过蒙特卡洛分析进行验证。我们通过评估三个现实世界的机器人：Pepper、G1 EDU和Digit来演示我们的方法。评分分析显示，各个平台的安全成熟度水平各不相同，评分范围从39.9%到79.5%不等。这项工作引入了一种结构化的、基于证据的评估方法，可以实现系统性的安全评估，支持跨平台基准测试，并指导人形机器人安全投资的优先顺序。



## **14. FRAME : Comprehensive Risk Assessment Framework for Adversarial Machine Learning Threats**

FRAME：对抗性机器学习威胁的全面风险评估框架 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17405v1) [paper-pdf](http://arxiv.org/pdf/2508.17405v1)

**Authors**: Avishag Shapira, Simon Shigol, Asaf Shabtai

**Abstract**: The widespread adoption of machine learning (ML) systems increased attention to their security and emergence of adversarial machine learning (AML) techniques that exploit fundamental vulnerabilities in ML systems, creating an urgent need for comprehensive risk assessment for ML-based systems. While traditional risk assessment frameworks evaluate conventional cybersecurity risks, they lack ability to address unique challenges posed by AML threats. Existing AML threat evaluation approaches focus primarily on technical attack robustness, overlooking crucial real-world factors like deployment environments, system dependencies, and attack feasibility. Attempts at comprehensive AML risk assessment have been limited to domain-specific solutions, preventing application across diverse systems. Addressing these limitations, we present FRAME, the first comprehensive and automated framework for assessing AML risks across diverse ML-based systems. FRAME includes a novel risk assessment method that quantifies AML risks by systematically evaluating three key dimensions: target system's deployment environment, characteristics of diverse AML techniques, and empirical insights from prior research. FRAME incorporates a feasibility scoring mechanism and LLM-based customization for system-specific assessments. Additionally, we developed a comprehensive structured dataset of AML attacks enabling context-aware risk assessment. From an engineering application perspective, FRAME delivers actionable results designed for direct use by system owners with only technical knowledge of their systems, without expertise in AML. We validated it across six diverse real-world applications. Our evaluation demonstrated exceptional accuracy and strong alignment with analysis by AML experts. FRAME enables organizations to prioritize AML risks, supporting secure AI deployment in real-world environments.

摘要: 机器学习（ML）系统的广泛采用增加了人们对其安全性的关注，以及利用ML系统基本漏洞的对抗性机器学习（ML）技术的出现，迫切需要对基于ML的系统进行全面风险评估。虽然传统的风险评估框架评估传统的网络安全风险，但它们缺乏应对反洗钱威胁带来的独特挑战的能力。现有的反洗钱威胁评估方法主要关注技术攻击的稳健性，而忽略了部署环境、系统依赖性和攻击可行性等关键现实世界因素。全面的反洗钱风险评估的尝试仅限于特定领域的解决方案，从而阻止了跨不同系统的应用。为了解决这些限制，我们提出了FRAME，这是第一个用于评估各种基于ML的系统中的反洗钱风险的全面自动化框架。FRAME包括一种新颖的风险评估方法，通过系统性评估三个关键维度来量化反洗钱风险：目标系统的部署环境、各种反洗钱技术的特征以及来自先前研究的经验见解。FRAME结合了可行性评分机制和基于LLM的定制，用于特定于系统的评估。此外，我们还开发了一个全面的结构化的反洗钱攻击数据集，从而实现了上下文感知的风险评估。从工程应用程序的角度来看，FRAME提供可操作的结果，供仅具有系统技术知识而不具备反洗钱专业知识的系统所有者直接使用。我们在六个不同的现实世界应用程序中验证了它。我们的评估证明了异常准确性，并且与反洗钱专家的分析高度一致。FRAME使组织能够优先考虑反洗钱风险，支持在现实世界环境中安全的人工智能部署。



## **15. Bridging Models to Defend: A Population-Based Strategy for Robust Adversarial Defense**

桥梁模型以防御：基于人口的稳健对抗防御策略 cs.AI

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2303.10225v2) [paper-pdf](http://arxiv.org/pdf/2303.10225v2)

**Authors**: Ren Wang, Yuxuan Li, Can Chen, Dakuo Wang, Jinjun Xiong, Pin-Yu Chen, Sijia Liu, Mohammad Shahidehpour, Alfred Hero

**Abstract**: Adversarial robustness is a critical measure of a neural network's ability to withstand adversarial attacks at inference time. While robust training techniques have improved defenses against individual $\ell_p$-norm attacks (e.g., $\ell_2$ or $\ell_\infty$), models remain vulnerable to diversified $\ell_p$ perturbations. To address this challenge, we propose a novel Robust Mode Connectivity (RMC)-oriented adversarial defense framework comprising two population-based learning phases. In Phase I, RMC searches the parameter space between two pre-trained models to construct a continuous path containing models with high robustness against multiple $\ell_p$ attacks. To improve efficiency, we introduce a Self-Robust Mode Connectivity (SRMC) module that accelerates endpoint generation in RMC. Building on RMC, Phase II presents RMC-based optimization, where RMC modules are composed to further enhance diversified robustness. To increase Phase II efficiency, we propose Efficient Robust Mode Connectivity (ERMC), which leverages $\ell_1$- and $\ell_\infty$-adversarially trained models to achieve robustness across a broad range of $p$-norms. An ensemble strategy is employed to further boost ERMC's performance. Extensive experiments across diverse datasets and architectures demonstrate that our methods significantly improve robustness against $\ell_\infty$, $\ell_2$, $\ell_1$, and hybrid attacks. Code is available at https://github.com/wangren09/MCGR.

摘要: 对抗鲁棒性是神经网络在推理时抵御对抗攻击能力的关键指标。虽然强大的训练技术改进了对个体$\ell_p$-norm攻击的防御（例如，$\ell_2 $或$\ell_\infty$），模型仍然容易受到多元化$\ell_p$扰动的影响。为了应对这一挑战，我们提出了一种新型的面向鲁棒模式连接性（RMC）的对抗防御框架，该框架包括两个基于群体的学习阶段。在第一阶段，RMC搜索两个预先训练好的模型之间的参数空间，以构建一个包含对多种$\ell_p$攻击具有高鲁棒性的模型的连续路径。为了提高效率，我们引入了一个自鲁棒模式连接（SRMC）模块，加快端点生成RMC。在RMC的基础上，第二阶段提出了基于RMC的优化，其中RMC模块的组成，以进一步提高多样化的鲁棒性。为了提高第二阶段的效率，我们提出了高效鲁棒模式连接（ERMC），它利用$\ell_1 $-和$\ell_\infty$-逆向训练模型来实现广泛的$p$-范数的鲁棒性。采用整体策略进一步提高ERMC的性能。跨不同数据集和架构的广泛实验表明，我们的方法显着提高了针对$\ell_\infty$、$\ell_2 $、$\ell_1 $和混合攻击的稳健性。代码可在https://github.com/wangren09/MCGR上获取。



## **16. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17361v1) [paper-pdf](http://arxiv.org/pdf/2508.17361v1)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅有效，而且可以跨模型（GPT-4 o、Claude 3.5、Gemini 2.0）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **17. Risk Assessment and Security Analysis of Large Language Models**

大型语言模型的风险评估与安全性分析 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17329v1) [paper-pdf](http://arxiv.org/pdf/2508.17329v1)

**Authors**: Xiaoyan Zhang, Dongyang Lyu, Xiaoqi Li

**Abstract**: As large language models (LLMs) expose systemic security challenges in high risk applications, including privacy leaks, bias amplification, and malicious abuse, there is an urgent need for a dynamic risk assessment and collaborative defence framework that covers their entire life cycle. This paper focuses on the security problems of large language models (LLMs) in critical application scenarios, such as the possibility of disclosure of user data, the deliberate input of harmful instructions, or the models bias. To solve these problems, we describe the design of a system for dynamic risk assessment and a hierarchical defence system that allows different levels of protection to cooperate. This paper presents a risk assessment system capable of evaluating both static and dynamic indicators simultaneously. It uses entropy weighting to calculate essential data, such as the frequency of sensitive words, whether the API call is typical, the realtime risk entropy value is significant, and the degree of context deviation. The experimental results show that the system is capable of identifying concealed attacks, such as role escape, and can perform rapid risk evaluation. The paper uses a hybrid model called BERT-CRF (Bidirectional Encoder Representation from Transformers) at the input layer to identify and filter malicious commands. The model layer uses dynamic adversarial training and differential privacy noise injection technology together. The output layer also has a neural watermarking system that can track the source of the content. In practice, the quality of this method, especially important in terms of customer service in the financial industry.

摘要: 由于大型语言模型（LLM）在高风险应用中暴露出系统性的安全挑战，包括隐私泄露，偏见放大和恶意滥用，因此迫切需要一个涵盖其整个生命周期的动态风险评估和协作防御框架。本文重点研究了大型语言模型在关键应用场景中的安全问题，如用户数据泄露的可能性、有害指令的故意输入、模型偏差等。为了解决这些问题，我们描述了一个系统的设计，动态风险评估和分级防御系统，允许不同级别的保护合作。本文提出了一种能够同时评估静态和动态指标的风险评估系统。它使用熵加权来计算基本数据，例如敏感词的频率、API调用是否典型、实时风险熵值是否重要以及上下文偏离程度。实验结果表明，该系统能够识别角色逃避等隐藏攻击，并能够进行快速风险评估。该论文在输入层使用了一种名为BERT-RF（来自Transformers的双向编码器表示）的混合模型来识别和过滤恶意命令。模型层结合使用动态对抗训练和差异隐私噪音注入技术。输出层还具有一个可以跟踪内容来源的神经水印系统。在实践中，这种方法的质量对于金融行业的客户服务尤其重要。



## **18. AdaGAT: Adaptive Guidance Adversarial Training for the Robustness of Deep Neural Networks**

AdaGAT：深度神经网络鲁棒性的自适应引导对抗训练 cs.CV

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17265v1) [paper-pdf](http://arxiv.org/pdf/2508.17265v1)

**Authors**: Zhenyu Liu, Huizhi Liang, Xinrun Li, Vaclav Snasel, Varun Ojha

**Abstract**: Adversarial distillation (AD) is a knowledge distillation technique that facilitates the transfer of robustness from teacher deep neural network (DNN) models to lightweight target (student) DNN models, enabling the target models to perform better than only training the student model independently. Some previous works focus on using a small, learnable teacher (guide) model to improve the robustness of a student model. Since a learnable guide model starts learning from scratch, maintaining its optimal state for effective knowledge transfer during co-training is challenging. Therefore, we propose a novel Adaptive Guidance Adversarial Training (AdaGAT) method. Our method, AdaGAT, dynamically adjusts the training state of the guide model to install robustness to the target model. Specifically, we develop two separate loss functions as part of the AdaGAT method, allowing the guide model to participate more actively in backpropagation to achieve its optimal state. We evaluated our approach via extensive experiments on three datasets: CIFAR-10, CIFAR-100, and TinyImageNet, using the WideResNet-34-10 model as the target model. Our observations reveal that appropriately adjusting the guide model within a certain accuracy range enhances the target model's robustness across various adversarial attacks compared to a variety of baseline models.

摘要: 对抗蒸馏（AD）是一种知识蒸馏技术，可促进鲁棒性从教师深度神经网络（DNN）模型转移到轻量级目标（学生）DNN模型，使目标模型比仅独立训练学生模型表现得更好。之前的一些作品专注于使用小型、可学习的教师（指导）模型来提高学生模型的稳健性。由于可学习的指导模型从零开始学习，因此在联合培训期间保持其有效知识转移的最佳状态具有挑战性。因此，我们提出了一种新型的自适应引导对抗训练（AdaGAT）方法。我们的方法AdaGAT动态调整引导模型的训练状态，以为目标模型提供鲁棒性。具体来说，我们开发了两个独立的损失函数，作为AdaGAT方法的一部分，允许引导模型更积极地参与反向传播以实现其最佳状态。我们使用WideResNet-34-10模型作为目标模型，通过对三个数据集（CIFAR-10、CIFAR-100和TinyImageNet）进行广泛实验来评估我们的方法。我们的观察表明，与各种基线模型相比，在一定的准确性范围内适当调整引导模型可以增强目标模型在各种对抗攻击中的鲁棒性。



## **19. Uncovering and Mitigating Destructive Multi-Embedding Attacks in Deepfake Proactive Forensics**

发现和缓解Deepfake主动取证中的破坏性多重嵌入攻击 cs.CV

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17247v1) [paper-pdf](http://arxiv.org/pdf/2508.17247v1)

**Authors**: Lixin Jia, Haiyang Sun, Zhiqing Guo, Yunfeng Diao, Dan Ma, Gaobo Yang

**Abstract**: With the rapid evolution of deepfake technologies and the wide dissemination of digital media, personal privacy is facing increasingly serious security threats. Deepfake proactive forensics, which involves embedding imperceptible watermarks to enable reliable source tracking, serves as a crucial defense against these threats. Although existing methods show strong forensic ability, they rely on an idealized assumption of single watermark embedding, which proves impractical in real-world scenarios. In this paper, we formally define and demonstrate the existence of Multi-Embedding Attacks (MEA) for the first time. When a previously protected image undergoes additional rounds of watermark embedding, the original forensic watermark can be destroyed or removed, rendering the entire proactive forensic mechanism ineffective. To address this vulnerability, we propose a general training paradigm named Adversarial Interference Simulation (AIS). Rather than modifying the network architecture, AIS explicitly simulates MEA scenarios during fine-tuning and introduces a resilience-driven loss function to enforce the learning of sparse and stable watermark representations. Our method enables the model to maintain the ability to extract the original watermark correctly even after a second embedding. Extensive experiments demonstrate that our plug-and-play AIS training paradigm significantly enhances the robustness of various existing methods against MEA.

摘要: 随着Deepfake技术的快速发展和数字媒体的广泛传播，个人隐私面临着日益严重的安全威胁。Deepfake主动取证涉及嵌入不可感知的水印以实现可靠的源跟踪，是抵御这些威胁的重要防御措施。尽管现有的方法显示出很强的取证能力，但它们依赖于单个水印嵌入的理想化假设，这在现实世界的场景中被证明是不切实际的。本文首次正式定义并证明了多重嵌入攻击（MEA）的存在性。当之前保护的图像经历额外几轮水印嵌入时，原始的取证水印可能会被破坏或删除，从而导致整个主动取证机制无效。为了解决这个漏洞，我们提出了一种名为对抗干扰模拟（AIS）的通用训练范式。AIS没有修改网络架构，而是在微调期间显式地模拟了多边环境协议（MTA）场景，并引入了顺从驱动的损失函数来强制学习稀疏和稳定的水印表示。我们的方法使模型即使在第二次嵌入之后也能够保持正确提取原始水印的能力。大量实验表明，我们的即插即用的AIS训练范式显着增强了各种现有方法针对MTA的鲁棒性。



## **20. How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System**

如何让医疗人工智能系统更安全？模拟多模式医疗RAG系统中的漏洞和威胁 cs.LG

Sumbitted to 2025 AAAI main track

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17215v1) [paper-pdf](http://arxiv.org/pdf/2508.17215v1)

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems.

摘要: 通过检索增强生成（RAG）增强的大型视觉语言模型（LVLM）越来越多地用于医疗人工智能，以通过外部临床图像文本检索增强事实基础。然而，这种依赖造成了重大的攻击面。我们提出了MedThreatRAG，这是一种新型的多模式中毒框架，通过注入对抗性图像-文本对来系统性地探索医疗RAG系统中的漏洞。我们方法的一个关键创新是构建模拟的半开放攻击环境，模仿现实世界的医疗系统，允许通过用户或管道贡献定期更新知识库。在此背景下，我们引入并强调跨模式冲突注入（CCGI），它嵌入了医学图像及其配对报告之间的微妙语义矛盾。这些不匹配通过扰乱跨模式对齐而降低检索和生成，同时保持足够合理以逃避传统过滤器。虽然为了完整性而包含了基本的文本和视觉攻击，但CMCI表现出了最严重的降级。对IU-X射线和MIIC-CXR QA任务的评估表明，MedThreatRAG将答案F1评分降低高达27.66%，并将LLaBA-Med-1.5 F1评分降低至低至51.36%。我们的研究结果揭示了临床RAG系统中的根本安全漏洞，并强调了对威胁感知设计和强大的多模式一致性检查的迫切需求。最后，我们提出了一套简洁的指南，为未来多模式医疗RAG系统的安全开发提供信息。



## **21. Adversarial Illusions in Multi-Modal Embeddings**

多模式嵌入中的对抗幻象 cs.CR

In USENIX Security'24

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2308.11804v5) [paper-pdf](http://arxiv.org/pdf/2308.11804v5)

**Authors**: Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode texts, images, thermal images, sounds, and videos into a single embedding space, aligning representations across different modalities (e.g., associate an image of a dog with a barking sound). In this paper, we show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it to make its embedding close to an arbitrary, adversary-chosen input in another modality.   These attacks are cross-modal and targeted: the adversary can align any image or sound with any target of his choice. Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks and modalities, enabling a wholesale compromise of current and future tasks, as well as modalities not available to the adversary. Using ImageBind and AudioCLIP embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, zero-shot classification, and audio retrieval.   We investigate transferability of illusions across different embeddings and develop a black-box version of our method that we use to demonstrate the first adversarial alignment attack on Amazon's commercial, proprietary Titan embedding. Finally, we analyze countermeasures and evasion attacks.

摘要: 多模式嵌入将文本、图像、热图像、声音和视频编码到单个嵌入空间中，将不同模式的表示对齐（例如，将狗的图像与吠叫的声音联系起来）。在本文中，我们表明多模式嵌入可能容易受到我们称之为“对抗错觉”的攻击。“给定图像或声音，对手可以扰乱它，使其嵌入接近另一种模式中任意的、对手选择的输入。   这些攻击是跨模式和有针对性的：对手可以将任何图像或声音与他选择的任何目标对齐。对抗幻象利用嵌入空间中的邻近性，因此对下游任务和模式不可知，从而能够对当前和未来任务以及对手无法使用的模式进行大规模妥协。使用Image Bind和AudioCLIP嵌入，我们演示了在不了解特定下游任务的情况下生成的反向对齐输入如何误导图像生成、文本生成、零镜头分类和音频检索。   我们研究错觉在不同嵌入中的可移植性，并开发我们方法的黑匣子版本，我们用它来演示对亚马逊商业专有泰坦嵌入的第一次对抗性对齐攻击。最后，我们分析了应对措施和规避攻击。



## **22. Sharpness-Aware Geometric Defense for Robust Out-Of-Distribution Detection**

用于鲁棒性分布外检测的敏锐度感知几何防御 cs.LG

under review

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17174v1) [paper-pdf](http://arxiv.org/pdf/2508.17174v1)

**Authors**: Jeng-Lin Li, Ming-Ching Chang, Wei-Chao Chen

**Abstract**: Out-of-distribution (OOD) detection ensures safe and reliable model deployment. Contemporary OOD algorithms using geometry projection can detect OOD or adversarial samples from clean in-distribution (ID) samples. However, this setting regards adversarial ID samples as OOD, leading to incorrect OOD predictions. Existing efforts on OOD detection with ID and OOD data under attacks are minimal. In this paper, we develop a robust OOD detection method that distinguishes adversarial ID samples from OOD ones. The sharp loss landscape created by adversarial training hinders model convergence, impacting the latent embedding quality for OOD score calculation. Therefore, we introduce a {\bf Sharpness-aware Geometric Defense (SaGD)} framework to smooth out the rugged adversarial loss landscape in the projected latent geometry. Enhanced geometric embedding convergence enables accurate ID data characterization, benefiting OOD detection against adversarial attacks. We use Jitter-based perturbation in adversarial training to extend the defense ability against unseen attacks. Our SaGD framework significantly improves FPR and AUC over the state-of-the-art defense approaches in differentiating CIFAR-100 from six other OOD datasets under various attacks. We further examine the effects of perturbations at various adversarial training levels, revealing the relationship between the sharp loss landscape and adversarial OOD detection.

摘要: 分发外（OOD）检测确保安全可靠的模型部署。使用几何投影的当代OOD算法可以从干净的内分布（ID）样本中检测OOD或对抗样本。然而，这种设置将对抗ID样本视为OOD，从而导致OOD预测错误。目前在攻击下使用ID和OOD数据进行OOD检测的工作很少。本文中，我们开发了一种强大的OOD检测方法，可以区分对抗性ID样本和OOD样本。对抗训练造成的急剧损失格局阻碍了模型收敛，影响了OOD分数计算的潜在嵌入质量。因此，我们引入了一个{\BF敏锐性几何防御（SaVD）}框架，以平滑投影潜在几何中崎岖的对抗损失景观。增强的几何嵌入融合可以实现准确的ID数据特征，有利于OOD检测对抗攻击。我们在对抗训练中使用基于抖动的扰动来扩展针对不可见攻击的防御能力。与最先进的防御方法相比，我们的SaVD框架显着提高了FPR和AUC，将CIFAR-100与各种攻击下的其他六个OOD数据集区分开来。我们进一步研究了不同对抗训练水平下扰动的影响，揭示了急剧损失景观和对抗性OOD检测之间的关系。



## **23. Towards Safeguarding LLM Fine-tuning APIs against Cipher Attacks**

保护LLM微调API免受密码攻击 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17158v1) [paper-pdf](http://arxiv.org/pdf/2508.17158v1)

**Authors**: Jack Youstra, Mohammed Mahfoud, Yang Yan, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: Large language model fine-tuning APIs enable widespread model customization, yet pose significant safety risks. Recent work shows that adversaries can exploit access to these APIs to bypass model safety mechanisms by encoding harmful content in seemingly harmless fine-tuning data, evading both human monitoring and standard content filters. We formalize the fine-tuning API defense problem, and introduce the Cipher Fine-tuning Robustness benchmark (CIFR), a benchmark for evaluating defense strategies' ability to retain model safety in the face of cipher-enabled attackers while achieving the desired level of fine-tuning functionality. We include diverse cipher encodings and families, with some kept exclusively in the test set to evaluate for generalization across unseen ciphers and cipher families. We then evaluate different defenses on the benchmark and train probe monitors on model internal activations from multiple fine-tunes. We show that probe monitors achieve over 99% detection accuracy, generalize to unseen cipher variants and families, and compare favorably to state-of-the-art monitoring approaches. We open-source CIFR and the code to reproduce our experiments to facilitate further research in this critical area. Code and data are available online https://github.com/JackYoustra/safe-finetuning-api

摘要: 大型语言模型微调API可以实现广泛的模型定制，但也会带来重大的安全风险。最近的工作表明，对手可以利用对这些API的访问来绕过模型安全机制，将有害内容编码在看似无害的微调数据中，从而逃避人类监控和标准内容过滤器。我们形式化了微调API防御问题，并引入了Cipher微调稳健性基准（CIFR），这是一个评估防御策略在面对启用密码的攻击者时保持模型安全性的能力的基准，同时实现了所需的微调功能水平。我们包括不同的密码编码和系列，其中一些仅保留在测试集中，以评估未见密码和密码系列的通用性。然后，我们评估基准上的不同防御，并根据多个微调的模型内部激活训练探测器监视器。我们表明，探针监测器实现了超过99%的检测准确率，可推广到未见的密码变体和家族，并且与最先进的监测方法相比具有优势。我们开源CIFR和复制我们实验的代码，以促进这一关键领域的进一步研究。代码和数据可在线获取https://github.com/JackYoustra/safe-finetuning-api



## **24. ZAPS: A Zero-Knowledge Proof Protocol for Secure UAV Authentication with Flight Path Privacy**

ZAPS：一种具有飞行路径隐私的安全无人机认证的零知识证明协议 cs.CR

11 Pages, 8 figures, Journal

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17043v1) [paper-pdf](http://arxiv.org/pdf/2508.17043v1)

**Authors**: Shayesta Naziri, Xu Wang, Guangsheng Yu, Christy Jie Liang, Wei Ni

**Abstract**: The increasing deployment of Unmanned Aerial Vehicles (UAVs) for military, commercial, and logistics applications has raised significant concerns regarding flight path privacy. Conventional UAV communication systems often expose flight path data to third parties, making them vulnerable to tracking, surveillance, and location inference attacks. Existing encryption techniques provide security but fail to ensure complete privacy, as adversaries can still infer movement patterns through metadata analysis. To address these challenges, we propose a zk-SNARK(Zero-Knowledge Succinct Non-Interactive Argument of Knowledge)-based privacy-preserving flight path authentication and verification framework. Our approach ensures that a UAV can prove its authorisation, validate its flight path with a control centre, and comply with regulatory constraints without revealing any sensitive trajectory information. By leveraging zk-SNARKs, the UAV can generate cryptographic proofs that verify compliance with predefined flight policies while keeping the exact path and location undisclosed. This method mitigates risks associated with real-time tracking, identity exposure, and unauthorised interception, thereby enhancing UAV operational security in adversarial environments. Our proposed solution balances privacy, security, and computational efficiency, making it suitable for resource-constrained UAVs in both civilian and military applications.

摘要: 越来越多的无人机（UAV）用于军事、商业和物流应用，引起了人们对飞行路径隐私的严重关注。传统的无人机通信系统经常将飞行路径数据暴露给第三方，使其容易受到跟踪，监视和位置推断攻击。现有的加密技术提供了安全性，但无法确保完全的隐私，因为对手仍然可以通过元数据分析推断移动模式。为了应对这些挑战，我们提出了一个基于zk-SNARK（Zero-Knowledge Succinct Non-Interactive Argument of Knowledge）的隐私保护飞行路径认证和验证框架。我们的方法确保无人机能够证明其授权，通过控制中心验证其飞行路径，并遵守监管限制，而不会泄露任何敏感轨迹信息。通过利用zk-SNARKS，无人机可以生成加密证据，验证是否符合预定义的飞行政策，同时保持确切的路径和位置不公开。该方法可以降低与实时跟踪、身份暴露和未经授权拦截相关的风险，从而增强无人机在对抗环境中的操作安全性。我们提出的解决方案平衡了隐私、安全和计算效率，使其适合民用和军事应用中的资源受限的无人机。



## **25. Watermarking Visual Concepts for Diffusion Models**

扩散模型的视觉概念水印 cs.CR

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2411.11688v3) [paper-pdf](http://arxiv.org/pdf/2411.11688v3)

**Authors**: Liangqi Lei, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: The personalization techniques of diffusion models succeed in generating images with specific concepts. This ability also poses great threats to copyright protection and network security since malicious users can generate unauthorized content and disinformation relevant to a target concept. Model watermarking is an effective solution to trace the malicious generated images and safeguard their copyright. However, existing model watermarking techniques merely achieve image-level tracing without concept traceability. When tracing infringing or harmful concepts, current approaches execute image concept detection and model tracing sequentially, where performance is critically constrained by concept detection accuracy. In this paper, we propose a lightweight concept watermarking framework that efficiently binds target concepts to model watermarks, supporting simultaneous concept identification and model tracing via single-stage watermark verification. To further enhance the robustness of concept watermarking, we propose an adversarial perturbation injection method collaboratively embedded with watermarks during image generation, avoiding watermark removal by model purification attacks. Experimental results demonstrate that ConceptWM significantly outperforms state-of-the-art watermarking methods, improving detection accuracy by 6.3%-19.3% across diverse datasets including COCO and StableDiffusionDB. Additionally, ConceptWM possesses a critical capability absent in other watermarking methods: it sustains a 21.7% FID/CLIP degradation under adversarial fine-tuning of Stable Diffusion models on WikiArt and CelebA-HQ, demonstrating its capability to mitigate model misuse.

摘要: 扩散模型的个性化技术成功地生成了具有特定概念的图像。这种能力还对版权保护和网络安全构成巨大威胁，因为恶意用户可以生成与目标概念相关的未经授权的内容和虚假信息。模型水印是追踪恶意生成图像并保护其版权的有效解决方案。然而，现有的模型水印技术仅实现图像级跟踪，而没有概念可追溯性。当跟踪侵权或有害概念时，当前的方法顺序执行图像概念检测和模型跟踪，其中性能受到概念检测准确性的严格限制。本文提出了一种轻量级概念水印框架，该框架有效地将目标概念与模型水印绑定，支持通过单阶段水印验证同时进行概念识别和模型跟踪。为了进一步增强概念水印的鲁棒性，我们提出了一种在图像生成过程中协同嵌入水印的对抗扰动注入方法，避免了模型净化攻击的水印去除。实验结果表明，ConceptWM的性能显着优于最先进的水印方法，在COCO和StableDiffusionDB等不同数据集中将检测准确率提高了6.3%-19.3%。此外，ConceptWM还拥有其他水印方法所不具备的关键能力：在对WikiArt和CelebA-HQ上的Stable Distance模型进行对抗性微调的情况下，它可以维持21.7%的DID/CLIP降级，证明了其减轻模型滥用的能力。



## **26. Unveiling the Latent Directions of Reflection in Large Language Models**

揭示大型语言模型中反射的潜在方向 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.16989v1) [paper-pdf](http://arxiv.org/pdf/2508.16989v1)

**Authors**: Fu-Chieh Chang, Yu-Ting Lee, Pei-Yuan Wu

**Abstract**: Reflection, the ability of large language models (LLMs) to evaluate and revise their own reasoning, has been widely used to improve performance on complex reasoning tasks. Yet, most prior work emphasizes designing reflective prompting strategies or reinforcement learning objectives, leaving the inner mechanisms of reflection underexplored. In this paper, we investigate reflection through the lens of latent directions in model activations. We propose a methodology based on activation steering to characterize how instructions with different reflective intentions: no reflection, intrinsic reflection, and triggered reflection. By constructing steering vectors between these reflection levels, we demonstrate that (1) new reflection-inducing instructions can be systematically identified, (2) reflective behavior can be directly enhanced or suppressed through activation interventions, and (3) suppressing reflection is considerably easier than stimulating it. Experiments on GSM8k-adv with Qwen2.5-3B and Gemma3-4B reveal clear stratification across reflection levels, and steering interventions confirm the controllability of reflection. Our findings highlight both opportunities (e.g., reflection-enhancing defenses) and risks (e.g., adversarial inhibition of reflection in jailbreak attacks). This work opens a path toward mechanistic understanding of reflective reasoning in LLMs.

摘要: 反射是大型语言模型（LLM）评估和修改自身推理的能力，已被广泛用于提高复杂推理任务的性能。然而，大多数先前的工作都强调设计反思性提示策略或强化学习目标，而反思的内部机制却没有得到充分的探索。本文中，我们研究了模型激活中潜在方向的镜头的反射。我们提出了一种基于激活引导的方法论来描述具有不同反射意图的指令：无反射、内在反射和触发反射。通过在这些反射水平之间构建引导载体，我们证明了（1）可以系统地识别新的反射诱导指令，（2）可以通过激活干预直接增强或抑制反射行为，（3）抑制反射比刺激反射容易得多。在GSM 8 k-adv上使用Qwen 2.5 -3B和Gemma 3 - 4 B进行的实验揭示了反射水平之间的明显分层，而引导干预证实了反思的可控性。我们的调查结果强调了这两种机会（例如，反思增强防御）和风险（例如，越狱攻击中反思的对抗性抑制）。这项工作开辟了对法学硕士中反思推理的机械理解的道路。



## **27. Effective Red-Teaming of Policy-Adherent Agents**

有效的红色团队的政策遵守代理 cs.MA

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2506.09600v3) [paper-pdf](http://arxiv.org/pdf/2506.09600v3)

**Authors**: Itay Nakash, George Kour, Koren Lazar, Matan Vetzler, Guy Uziel, Ateret Anaby-Tavor

**Abstract**: Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks

摘要: 以任务为导向的基于LLM的代理越来越多地用于具有严格政策（例如退款资格或取消规则）的领域。挑战在于确保代理始终遵守这些规则和政策，适当地拒绝任何可能违反这些规则和政策的请求，同时仍然保持有用和自然的交互。这就要求开发量身定制的设计和评估方法，以确保代理对恶意用户行为的弹性。我们提出了一种新的威胁模型，专注于对抗性的用户，旨在利用遵守政策的代理人为个人利益。为了解决这个问题，我们提出了CRAFT，这是一个多代理红色团队系统，它利用政策感知的说服策略来破坏客户服务场景中遵守政策的代理，优于传统的越狱方法，例如DAN提示、情绪操纵和胁迫。在现有的tau-table基准的基础上，我们引入了tau-break，这是一个补充基准，旨在严格评估代理针对操纵用户行为的稳健性。最后，我们评估了几种简单但有效的防御策略。虽然这些措施提供了一些保护，但它们还不够，这凸显了需要更强大的、以研究为驱动的保障措施来保护遵守政策的代理人免受对抗性攻击



## **28. NAT: Learning to Attack Neurons for Enhanced Adversarial Transferability**

纳特：学习攻击神经元以增强对抗可移植性 cs.CV

Published at WACV 2025

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.16937v1) [paper-pdf](http://arxiv.org/pdf/2508.16937v1)

**Authors**: Krishna Kanth Nakka, Alexandre Alahi

**Abstract**: The generation of transferable adversarial perturbations typically involves training a generator to maximize embedding separation between clean and adversarial images at a single mid-layer of a source model. In this work, we build on this approach and introduce Neuron Attack for Transferability (NAT), a method designed to target specific neuron within the embedding. Our approach is motivated by the observation that previous layer-level optimizations often disproportionately focus on a few neurons representing similar concepts, leaving other neurons within the attacked layer minimally affected. NAT shifts the focus from embedding-level separation to a more fundamental, neuron-specific approach. We find that targeting individual neurons effectively disrupts the core units of the neural network, providing a common basis for transferability across different models. Through extensive experiments on 41 diverse ImageNet models and 9 fine-grained models, NAT achieves fooling rates that surpass existing baselines by over 14\% in cross-model and 4\% in cross-domain settings. Furthermore, by leveraging the complementary attacking capabilities of the trained generators, we achieve impressive fooling rates within just 10 queries. Our code is available at: https://krishnakanthnakka.github.io/NAT/

摘要: 可转移对抗性扰动的生成通常涉及训练生成器，以最大化源模型的单个中间层上干净图像和对抗性图像之间的嵌入分离。在这项工作中，我们以这种方法为基础，引入了神经元可移植性攻击（Naturon Attack for Transferability）（纳特），这是一种旨在针对嵌入内特定神经元的方法。我们的方法的动机是这样一个观察，即之前的层级优化通常不成比例地关注代表相似概念的一些神经元，而受攻击层内的其他神经元受到的影响最小。纳特将重点从嵌入级分离转移到更基本的、特定于神经元的方法。我们发现，针对单个神经元有效地扰乱了神经网络的核心单元，为不同模型之间的可移植性提供了共同的基础。通过对41个不同ImageNet模型和9个细粒度模型的广泛实验，纳特实现了跨模型中超越现有基线的欺骗率超过现有基线14%以上，跨域设置中超越现有基线4%。此外，通过利用经过训练的生成器的补充攻击能力，我们在短短10个查询内就实现了令人印象深刻的愚弄率。我们的代码可访问：https://krishnakanthnakka.github.io/NAT/



## **29. Mitigating Jailbreaks with Intent-Aware LLMs**

利用意图意识的法学硕士缓解越狱 cs.CR

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.12072v2) [paper-pdf](http://arxiv.org/pdf/2508.12072v2)

**Authors**: Wei Jie Yeo, Ranjan Satapathy, Erik Cambria

**Abstract**: Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses. We publicly release our code at https://github.com/wj210/Intent_Jailbreak.

摘要: 尽管进行了广泛的安全调整，大型语言模型（LLM）仍然容易受到通过敌对设计的指令的越狱攻击，这反映了安全性和任务性能之间的持续权衡。在这项工作中，我们提出了Intent-FT，这是一种简单且轻量级的微调方法，它在响应之前显式训练LLM推断指令的潜在意图。通过对目标对抗指令集进行微调，Intent-FT使LLM能够将意图演绎推广到不可见的攻击，从而大幅提高其稳健性。我们全面评估开源和专有模型中的参数和非参数攻击，考虑攻击的危害性、效用、过度拒绝以及对白盒威胁的影响。从经验上看，Intent-FT始终如一地减轻了所有评估的攻击类别，没有一次攻击的成功率超过50%，而现有的防御措施仅保持部分有效。重要的是，我们的方法保留了模型的一般功能，并减少了对包含表面有害关键词的良性指令的过度拒绝。此外，使用Intent-FT训练的模型可以准确识别对抗性攻击中隐藏的有害意图，并且可以有效地转移这些习得的意图以增强普通模型防御。我们在https://github.com/wj210/Intent_Jailbreak上公开发布我们的代码。



## **30. Two-Level Priority Coding for Resilience to Arbitrary Blockage Patterns**

针对任意阻塞模式的弹性的两级优先级编码 cs.IT

Extended version of the paper accepted at IEEE Military  Communications Conference (MILCOM), 2025

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.16899v1) [paper-pdf](http://arxiv.org/pdf/2508.16899v1)

**Authors**: Mine Gokce Dogan, Abhiram Kadiyala, Jaimin Shah, Martina Cardone, Christina Fragouli

**Abstract**: Ultra-reliable low-latency communication is essential in mission-critical settings, including military applications, where persistent and asymmetric link blockages caused by mobility, jamming, or adversarial attacks can disrupt delay-sensitive transmissions. This paper addresses this challenge by deploying a multilevel diversity coding (MDC) scheme that controls the received information, offers distinct reliability guarantees based on the priority of data streams, and maintains low design and operational complexity as the number of network paths increases. For two priority levels over three edge-disjoint paths, the complete capacity region is characterized, showing that superposition coding achieves the region in general, whereas network coding is required only in a specific corner case. Moreover, sufficient conditions under which a simple superposition coding scheme achieves the capacity for an arbitrary number of paths are identified. To prove these results and provide a unified analytical framework, the problem of designing high-performing MDC schemes is shown to be equivalent to the problem of designing high-performing encoding schemes over a class of broadcast networks, referred to as combination networks in the literature.

摘要: 超可靠的低延迟通信在关键任务环境中至关重要，包括军事应用，其中由移动性、干扰或对抗性攻击引起的持续且不对称的链路阻塞可能会扰乱延迟敏感的传输。本文通过部署多层多样性编码（MDC）方案来解决这一挑战，该方案控制接收的信息，根据数据流的优先级提供独特的可靠性保证，并随着网络路径数量的增加而保持较低的设计和操作复杂性。对于三个边缘不相交路径上的两个优先级，描述了完整的容量区域，表明叠加编码通常可以实现该区域，而网络编码仅在特定的拐角情况下才需要。此外，确定了简单叠加编码方案实现任意数量路径容量的充分条件。为了证明这些结果并提供统一的分析框架，设计高性能MDC方案的问题被证明相当于在一类广播网络（在文献中称为组合网络）上设计高性能编码方案的问题。



## **31. ImF: Implicit Fingerprint for Large Language Models**

ImF：大型语言模型的隐式指纹 cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2503.21805v3) [paper-pdf](http://arxiv.org/pdf/2503.21805v3)

**Authors**: Jiaxuan Wu, Wanli Peng, Hang Fu, Yiming Xue, Juan Wen

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.

摘要: 训练大型语言模型（LLM）是资源密集型且昂贵的，因此保护LLM的知识产权（IP）至关重要。最近，将指纹嵌入LLM已成为建立模型所有权的流行方法。然而，现有的指纹识别技术通常嵌入具有弱语义一致性的可识别模式，导致指纹与LLM固有的自然问答（QA）行为显着不同。这种差异削弱了嵌入指纹的隐蔽性，并使它们容易受到对抗攻击。在本文中，我们首先通过引入一种名为世代修订干预（GRI）攻击的新型对抗攻击来证明现有指纹嵌入方法的关键漏洞。GRI攻击利用了当前指纹识别方法的语义脆弱性，通过破坏指纹弱相关的语义结构来有效地擦除指纹。我们的经验评估强调，传统的指纹识别方法受到GRI攻击的严重损害，揭示了其在现实对抗条件下稳健性的严重局限性。为了推进模型指纹识别的最新水平，我们提出了一种新型模型指纹范式，称为隐式指纹（ImF）。ImF利用隐写技术将所有权信息巧妙地嵌入自然文本中，随后使用思想链（CoT）提示构建语义连贯且上下文自然的QA对。这种设计确保指纹与标准模型行为无缝集成，与常规输出保持无区别，并大幅降低意外触发和有针对性删除的风险。我们对15个不同的LLM进行了ImF的全面评估，涵盖不同的架构和不同的规模。



## **32. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

语音认证和反欺骗系统威胁调查 cs.CR

This paper will be submitted to the Computer Science Review

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16843v1) [paper-pdf](http://arxiv.org/pdf/2508.16843v1)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.

摘要: 语音认证发生了重大变化，从依赖手工声学特征的传统系统到可以提取稳健的说话者嵌入的深度学习模型。这一进步扩大了其在金融、智能设备、执法等领域的应用。然而，随着采用率的增加，威胁也随之增加。本调查全面回顾了针对语音认证系统（PAS）和反欺骗对策（CM）的现代威胁格局，包括数据中毒、对抗性、深度伪造和对抗性欺骗攻击。我们按时间顺序追踪语音认证的发展，并研究漏洞如何随着技术进步而演变。对于每种类型的攻击，我们总结了方法论，强调常用的数据集，比较性能和局限性，并使用广泛接受的分类法组织现有文献。通过强调新出现的风险和公开挑战，本调查旨在支持开发更安全、更有弹性的语音认证系统。



## **33. DeMem: Privacy-Enhanced Robust Adversarial Learning via De-Memorization**

DeMem：通过去伪化的隐私增强鲁棒对抗学习 cs.LG

10 pages, Accepted at MLSP 2025

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2412.05767v3) [paper-pdf](http://arxiv.org/pdf/2412.05767v3)

**Authors**: Xiaoyu Luo, Qiongxiu Li

**Abstract**: Adversarial robustness, the ability of a model to withstand manipulated inputs that cause errors, is essential for ensuring the trustworthiness of machine learning models in real-world applications. However, previous studies have shown that enhancing adversarial robustness through adversarial training increases vulnerability to privacy attacks. While differential privacy can mitigate these attacks, it often compromises robustness against both natural and adversarial samples. Our analysis reveals that differential privacy disproportionately impacts low-risk samples, causing an unintended performance drop. To address this, we propose DeMem, which selectively targets high-risk samples, achieving a better balance between privacy protection and model robustness. DeMem is versatile and can be seamlessly integrated into various adversarial training techniques. Extensive evaluations across multiple training methods and datasets demonstrate that DeMem significantly reduces privacy leakage while maintaining robustness against both natural and adversarial samples. These results confirm DeMem's effectiveness and broad applicability in enhancing privacy without compromising robustness.

摘要: 对抗鲁棒性，即模型承受导致错误的操纵输入的能力，对于确保机器学习模型在现实世界应用中的可信度至关重要。然而，之前的研究表明，通过对抗训练增强对抗稳健性会增加隐私攻击的脆弱性。虽然差异隐私可以减轻这些攻击，但它通常会损害针对自然样本和对抗样本的稳健性。我们的分析表明，差异隐私对低风险样本的影响尤为严重，导致意外的性能下降。为了解决这个问题，我们提出了DeMem，它有选择地针对高风险样本，在隐私保护和模型稳健性之间实现更好的平衡。DeMem功能广泛，可以无缝集成到各种对抗训练技术中。对多种训练方法和数据集的广泛评估表明，DeMem显着减少了隐私泄露，同时保持了针对自然样本和对抗样本的稳健性。这些结果证实了DeMem在增强隐私而不损害稳健性方面的有效性和广泛适用性。



## **34. A Curious Case of Remarkable Resilience to Gradient Attacks via Fully Convolutional and Differentiable Front End with a Skip Connection**

通过具有跳过连接的完全卷积和差异前端对梯度攻击具有显着弹性的奇怪案例 cs.LG

Accepted at TMLR (2025/08)

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2402.17018v2) [paper-pdf](http://arxiv.org/pdf/2402.17018v2)

**Authors**: Leonid Boytsov, Ameya Joshi, Filipe Condessa

**Abstract**: We experimented with front-end enhanced neural models where a differentiable and fully convolutional model with a skip connection is added before a frozen backbone classifier. By training such composite models using a small learning rate for about one epoch, we obtained models that retained the accuracy of the backbone classifier while being unusually resistant to gradient attacks-including APGD and FAB-T attacks from the AutoAttack package-which we attribute to gradient masking. Although gradient masking is not new, the degree we observe is striking for fully differentiable models without obvious gradient-shattering-e.g., JPEG compression-or gradient-diminishing components.   The training recipe to produce such models is also remarkably stable and reproducible: We applied it to three datasets (CIFAR10, CIFAR100, and ImageNet) and several modern architectures (including vision Transformers) without a single failure case. While black-box attacks such as the SQUARE attack and zero-order PGD can partially overcome gradient masking, these attacks are easily defeated by simple randomized ensembles. We estimate that these ensembles achieve near-SOTA AutoAttack accuracy on CIFAR10, CIFAR100, and ImageNet (while retaining almost all clean accuracy of the original classifiers) despite having near-zero accuracy under adaptive attacks.   Adversarially training the backbone further amplifies this front-end "robustness". On CIFAR10, the respective randomized ensemble achieved 90.8$\pm 2.5\%$ (99\% CI) accuracy under the full AutoAttack while having only 18.2$\pm 3.6\%$ accuracy under the adaptive attack ($\varepsilon=8/255$, $L^\infty$ norm). We conclude the paper with a discussion of whether randomized ensembling can serve as a practical defense.   Code and instructions to reproduce key results are available. https://github.com/searchivarius/curious_case_of_gradient_masking

摘要: 我们实验了前端增强型神经模型，其中在冻结主干分类器之前添加了具有跳过连接的可微和完全卷积模型。通过在大约一个时期内使用较小的学习率训练此类复合模型，我们获得的模型保留了主干分类器的准确性，同时对梯度攻击（包括来自AutoAttack包的APGD和FAB-T攻击）具有异常抵抗力-我们将其归因于梯度掩蔽。尽管梯度掩蔽并不新鲜，但我们观察到的程度对于没有明显梯度破坏的完全可微模型来说是惊人的-例如，JPEG压缩或梯度递减组件。   生成此类模型的训练配方也非常稳定和可重复：我们将其应用于三个数据集（CIFAR 10、CIFAR 100和ImageNet）和几个现代架构（包括视觉变形金刚），没有出现任何故障案例。虽然SQUARE攻击和零阶PVD等黑匣子攻击可以部分克服梯度掩蔽，但这些攻击很容易被简单的随机集合击败。我们估计，尽管在自适应攻击下的准确性接近于零，但这些集成在CIFAR 10、CIFAR 100和ImageNet上实现了接近SOTA的AutoAttack准确性（同时保留了原始分类器的几乎所有准确性）。   对主干进行对抗训练进一步增强了这种前端“稳健性”。在CIFAR 10上，相应的随机化集成在完全AutoAttack下实现了90.8 $\PM 2.5\%$（99\% CI）的准确性，而在自适应攻击下仅实现了18.2 $\PM 3.6\%$准确性（$\varepð =8/255$，$L &\infty$ norm）。我们通过讨论随机集合是否可以作为实际防御来结束论文。   提供了重现关键结果的代码和说明。https://github.com/searchivarius/curious_case_of_gradient_masking



## **35. HAMSA: Hijacking Aligned Compact Models via Stealthy Automation**

HAMSA：通过隐形自动化劫持对齐的紧凑型车型 cs.CL

9 pages, 1 figure; article under review

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16484v1) [paper-pdf](http://arxiv.org/pdf/2508.16484v1)

**Authors**: Alexey Krylov, Iskander Vagizov, Dmitrii Korzh, Maryam Douiba, Azidine Guezzaz, Vladimir Kokh, Sergey D. Erokhin, Elena V. Tutubalina, Oleg Y. Rogov

**Abstract**: Large Language Models (LLMs), especially their compact efficiency-oriented variants, remain susceptible to jailbreak attacks that can elicit harmful outputs despite extensive alignment efforts. Existing adversarial prompt generation techniques often rely on manual engineering or rudimentary obfuscation, producing low-quality or incoherent text that is easily flagged by perplexity-based filters. We present an automated red-teaming framework that evolves semantically meaningful and stealthy jailbreak prompts for aligned compact LLMs. The approach employs a multi-stage evolutionary search, where candidate prompts are iteratively refined using a population-based strategy augmented with temperature-controlled variability to balance exploration and coherence preservation. This enables the systematic discovery of prompts capable of bypassing alignment safeguards while maintaining natural language fluency. We evaluate our method on benchmarks in English (In-The-Wild Jailbreak Prompts on LLMs), and a newly curated Arabic one derived from In-The-Wild Jailbreak Prompts on LLMs and annotated by native Arabic linguists, enabling multilingual assessment.

摘要: 大型语言模型（LLM），尤其是其紧凑的、以效率为导向的变体，仍然容易受到越狱攻击，尽管进行了广泛的对齐工作，这些攻击可能会引发有害的输出。现有的对抗性提示生成技术通常依赖于手动工程或基本混淆，从而产生低质量或不连贯的文本，这些文本很容易被基于困惑的过滤器标记。我们提出了一个自动化的红色团队框架，该框架进化出具有语义意义且隐蔽的越狱提示，以实现对齐的紧凑型LLM。该方法采用多阶段进化搜索，其中使用基于种群的策略迭代细化候选提示，并增强温度控制的变异性，以平衡探索和一致性保留。这使得系统性地发现能够绕过对齐保障措施，同时保持自然语言流利性的提示。我们用英语基准评估我们的方法（LLM上的In-The-Wild Jailbreak Pretts），以及一种新策划的阿拉伯语基准评估方法，该方法源自LLM上的In-The-Wild Jailbreak Pretts，并由母语阿拉伯语语言学家注释，从而实现多语言评估。



## **36. Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms**

基准测试系统对不利伤害的鲁棒性 cs.LG

52 Pages

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16481v1) [paper-pdf](http://arxiv.org/pdf/2508.16481v1)

**Authors**: Jonathan Nöther, Adish Singla, Goran Radanovic

**Abstract**: Ensuring the safe use of agentic systems requires a thorough understanding of the range of malicious behaviors these systems may exhibit when under attack. In this paper, we evaluate the robustness of LLM-based agentic systems against attacks that aim to elicit harmful actions from agents. To this end, we propose a novel taxonomy of harms for agentic systems and a novel benchmark, BAD-ACTS, for studying the security of agentic systems with respect to a wide range of harmful actions. BAD-ACTS consists of 4 implementations of agentic systems in distinct application environments, as well as a dataset of 188 high-quality examples of harmful actions. This enables a comprehensive study of the robustness of agentic systems across a wide range of categories of harmful behaviors, available tools, and inter-agent communication structures. Using this benchmark, we analyze the robustness of agentic systems against an attacker that controls one of the agents in the system and aims to manipulate other agents to execute a harmful target action. Our results show that the attack has a high success rate, demonstrating that even a single adversarial agent within the system can have a significant impact on the security. This attack remains effective even when agents use a simple prompting-based defense strategy. However, we additionally propose a more effective defense based on message monitoring. We believe that this benchmark provides a diverse testbed for the security research of agentic systems. The benchmark can be found at github.com/JNoether/BAD-ACTS

摘要: 确保代理系统的安全使用需要彻底了解这些系统在受到攻击时可能表现出的恶意行为范围。在本文中，我们评估了基于LLM的代理系统针对旨在引发代理有害行为的攻击的稳健性。为此，我们提出了一种新型的代理系统危害分类法和一种新型基准BAD-SYS，用于研究代理系统针对广泛有害行为的安全性。BAD-SYS由不同应用环境中的4个代理系统实现以及包含188个有害行为高质量示例的数据集组成。这使得能够对各种有害行为、可用工具和代理间通信结构的代理系统的稳健性进行全面研究。使用这个基准，我们分析了代理系统针对控制系统中一个代理并旨在操纵其他代理执行有害目标动作的攻击者的稳健性。我们的结果表明，攻击的成功率很高，这表明即使是系统内的单个对抗代理也会对安全性产生重大影响。即使代理使用简单的基于预算的防御策略，此攻击仍然有效。然而，我们还提出了一个更有效的防御基于消息监控。我们相信，这个基准提供了一个多样化的测试平台的安全性研究的代理系统。基准测试可以在github.com/JNoether/BAD-ACTS上找到



## **37. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

MCP-Guard：大型语言模型应用中模型上下文协议完整性的防御框架 cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.10991v2) [paper-pdf](http://arxiv.org/pdf/2508.10991v2)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.

摘要: 通过模型上下文协议（HCP）等协议将大型语言模型（LLM）与外部工具集成会引入严重的安全漏洞，包括提示注入、数据溢出和其他威胁。为了应对这些挑战，我们提出了MCP-Guard，这是一种专为LLM工具交互而设计的稳健、分层的防御架构。MCP-Guard采用三阶段检测管道，平衡效率与准确性：它从针对明显威胁的轻量级静态扫描和针对语义攻击的深度神经检测器，发展到我们微调的基于E5的模型，在识别对抗性提示方面实现了（96.01）的准确性。最后，轻量级LLM仲裁器合成这些信号以做出最终决策，同时最大限度地减少误报。为了促进严格的培训和评估，我们还引入了MCP-AttackBench，这是一个包含超过70，000个样本的综合基准。MCP-AttackBench源自公共数据集，并通过GPT-4进行增强，以HCP格式模拟不同的现实世界攻击载体，为未来研究保护LLM工具生态系统提供基础。



## **38. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

PAR-AdvGAN：通过渐进式自回归AdvGAN提高对抗攻击能力 cs.LG

Best paper award of ECML-PKDD 2025

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2502.12207v4) [paper-pdf](http://arxiv.org/pdf/2502.12207v4)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://github.com/LMBTough/PAR

摘要: 深度神经网络在各个领域都表现出了卓越的性能。然而，它们容易受到对抗性例子的影响，这可能导致错误的预测。生成对抗网络（GAN）可以利用生成器和鉴别器模型快速生成高质量的对抗示例。由于两个模块都以竞争和同步的方式进行训练，因此与传统方法相比，基于GAN的算法（如AdvGAN）可以生成具有更好可移植性的对抗性示例。然而，扰动的产生通常仅限于单次迭代，从而阻止这些示例充分利用方法的潜力。为了解决这个问题，我们引入了一种名为渐进式自动回归AdvGAN（PAR-AdvGAN）的新颖方法。它在渐进生成网络中集成了自回归迭代机制，以制作具有增强攻击能力的对抗性示例。我们通过大规模实验彻底评估了我们的PAR-AdvGAN方法，证明了其优于各种最先进的黑匣子对抗攻击以及原始的AdvGAN的性能。此外，PAR-AdvGAN显着加速了对抗性示例的生成，即Inception-v3模型上的速度高达每秒335.5帧，优于基于梯度的可转移攻击算法。我们的代码可访问：https://github.com/LMBTough/PAR



## **39. from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors**

来自良性进口有毒：通过对抗性隐喻越狱语言模型 cs.CL

arXiv admin note: substantial text overlap with arXiv:2412.12145

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2503.00038v4) [paper-pdf](http://arxiv.org/pdf/2503.00038v4)

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Jiangyu Lei, Qi Li

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.

摘要: 当前的研究揭示了大型语言模型（LLM）通过越狱攻击生成有害内容的风险。然而，他们忽视了从头开始直接产生有害内容比诱导LLM将良性内容校准为有害形式更困难。在我们的研究中，我们引入了一种新颖的攻击框架，该框架利用AdVersArial meTAphoR（AVATAR）来诱导LLM校准用于越狱的恶意隐喻。具体来说，为了回答有害查询，AVATAR自适应地识别一组良性但逻辑相关的隐喻作为初始种子。然后，在这些隐喻的驱动下，目标LLM被诱导对隐喻内容进行推理和校准，从而通过直接输出有害响应或校准隐喻和专业有害内容之间的残留来越狱。实验结果表明，AVATAR可以有效且可转移的越狱LLM，并在多个高级LLM之间实现最先进的攻击成功率。



## **40. Robustness of deep learning classification to adversarial input on GPUs: asynchronous parallel accumulation is a source of vulnerability**

深度学习分类对图形处理器上对抗输入的鲁棒性：同步并行积累是脆弱性的根源 cs.LG

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2503.17173v2) [paper-pdf](http://arxiv.org/pdf/2503.17173v2)

**Authors**: Sanjif Shanmugavelu, Mathieu Taillefumier, Christopher Culver, Vijay Ganesh, Oscar Hernandez, Ada Sedova

**Abstract**: The ability of machine learning (ML) classification models to resist small, targeted input perturbations -- known as adversarial attacks -- is a key measure of their safety and reliability. We show that floating-point non-associativity (FPNA) coupled with asynchronous parallel programming on GPUs is sufficient to result in misclassification, without any perturbation to the input. Additionally, we show that standard adversarial robustness results may be overestimated up to 4.6 when not considering machine-level details. We develop a novel black-box attack using Bayesian optimization to discover external workloads that can change the instruction scheduling which bias the output of reductions on GPUs and reliably lead to misclassification. Motivated by these results, we present a new learnable permutation (LP) gradient-based approach to learning floating-point operation orderings that lead to misclassifications. The LP approach provides a worst-case estimate in a computationally efficient manner, avoiding the need to run identical experiments tens of thousands of times over a potentially large set of possible GPU states or architectures. Finally, using instrumentation-based testing, we investigate parallel reduction ordering across different GPU architectures under external background workloads, when utilizing multi-GPU virtualization, and when applying power capping. Our results demonstrate that parallel reduction ordering varies significantly across architectures under the first two conditions, substantially increasing the search space required to fully test the effects of this parallel scheduler-based vulnerability. These results and the methods developed here can help to include machine-level considerations into adversarial robustness assessments, which can make a difference in safety and mission critical applications.

摘要: 机器学习（ML）分类模型抵抗小的、有针对性的输入扰动（称为对抗性攻击）的能力是衡量其安全性和可靠性的关键指标。我们表明，浮点非结合性（FPNA）与图形处理器上的同步并行编程相结合足以导致误分类，而不会对输入产生任何干扰。此外，我们表明，当不考虑机器级细节时，标准对抗鲁棒性结果可能会被高估高达4.6。我们使用Bayesian优化开发了一种新型的黑匣子攻击，以发现可以改变指令调度的外部工作负载，从而偏向于对图形处理器的简化的输出，并可靠地导致错误分类。受这些结果的启发，我们提出了一种新的可学习排列（LP）基于梯度的方法来学习导致错误分类的浮点运算顺序。LP方法以计算高效的方式提供了最坏情况的估计，从而避免了在可能大量的可能的图形处理器状态或架构上运行数万次相同的实验的需要。最后，使用基于仪器的测试，我们研究了外部背景工作负载下、利用多图形处理器虚拟化以及应用功率上限时不同图形处理器架构之间的并行约简排序。我们的结果表明，在前两个条件下，并行约简排序在不同架构中存在显着差异，从而大大增加了全面测试这种并行基于供应商的漏洞的影响所需的搜索空间。这些结果和这里开发的方法可以帮助将机器级的考虑因素纳入对抗稳健性评估中，这可以在安全性和关键任务应用中发挥作用。



## **41. An Investigation of Visual Foundation Models Robustness**

视觉基础模型鲁棒性研究 cs.CV

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16225v1) [paper-pdf](http://arxiv.org/pdf/2508.16225v1)

**Authors**: Sandeep Gupta, Roberto Passerone

**Abstract**: Visual Foundation Models (VFMs) are becoming ubiquitous in computer vision, powering systems for diverse tasks such as object detection, image classification, segmentation, pose estimation, and motion tracking. VFMs are capitalizing on seminal innovations in deep learning models, such as LeNet-5, AlexNet, ResNet, VGGNet, InceptionNet, DenseNet, YOLO, and ViT, to deliver superior performance across a range of critical computer vision applications. These include security-sensitive domains like biometric verification, autonomous vehicle perception, and medical image analysis, where robustness is essential to fostering trust between technology and the end-users. This article investigates network robustness requirements crucial in computer vision systems to adapt effectively to dynamic environments influenced by factors such as lighting, weather conditions, and sensor characteristics. We examine the prevalent empirical defenses and robust training employed to enhance vision network robustness against real-world challenges such as distributional shifts, noisy and spatially distorted inputs, and adversarial attacks. Subsequently, we provide a comprehensive analysis of the challenges associated with these defense mechanisms, including network properties and components to guide ablation studies and benchmarking metrics to evaluate network robustness.

摘要: 视觉基础模型（VFM）在计算机视觉中变得无处不在，为目标检测、图像分类、分割、姿态估计和运动跟踪等各种任务的系统提供动力。VFM正在利用深度学习模型的开创性创新，例如LeNet-5、AlexNet、ResNet、VGGNet、InceptionNet、DenseNet、YOLO和ViT，在一系列关键计算机视觉应用程序中提供卓越的性能。其中包括生物识别验证、自动驾驶车辆感知和医学图像分析等安全敏感领域，其中稳健性对于促进技术和最终用户之间的信任至关重要。本文研究了计算机视觉系统中至关重要的网络鲁棒性要求，以有效适应受照明、天气条件和传感器特征等因素影响的动态环境。我们研究了用于增强视觉网络稳健性的普遍经验防御和稳健训练，以对抗现实世界挑战，例如分布变化、噪音和空间失真的输入以及对抗性攻击。随后，我们对与这些防御机制相关的挑战进行了全面分析，包括指导消融研究的网络属性和组件以及评估网络稳健性的基准指标。



## **42. PromptFlare: Prompt-Generalized Defense via Cross-Attention Decoy in Diffusion-Based Inpainting**

AtlantFlare：基于扩散的修复中通过交叉注意诱饵的预算广义防御 cs.CV

Accepted to ACM MM 2025

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16217v1) [paper-pdf](http://arxiv.org/pdf/2508.16217v1)

**Authors**: Hohyun Na, Seunghoo Hong, Simon S. Woo

**Abstract**: The success of diffusion models has enabled effortless, high-quality image modifications that precisely align with users' intentions, thereby raising concerns about their potential misuse by malicious actors. Previous studies have attempted to mitigate such misuse through adversarial attacks. However, these approaches heavily rely on image-level inconsistencies, which pose fundamental limitations in addressing the influence of textual prompts. In this paper, we propose PromptFlare, a novel adversarial protection method designed to protect images from malicious modifications facilitated by diffusion-based inpainting models. Our approach leverages the cross-attention mechanism to exploit the intrinsic properties of prompt embeddings. Specifically, we identify and target shared token of prompts that is invariant and semantically uninformative, injecting adversarial noise to suppress the sampling process. The injected noise acts as a cross-attention decoy, diverting the model's focus away from meaningful prompt-image alignments and thereby neutralizing the effect of prompt. Extensive experiments on the EditBench dataset demonstrate that our method achieves state-of-the-art performance across various metrics while significantly reducing computational overhead and GPU memory usage. These findings highlight PromptFlare as a robust and efficient protection against unauthorized image manipulations. The code is available at https://github.com/NAHOHYUN-SKKU/PromptFlare.

摘要: 扩散模型的成功使得可以轻松、高质量的图像修改能够精确地符合用户意图，从而引发了人们对其可能被恶意行为者滥用的担忧。之前的研究试图通过对抗攻击来减轻这种滥用。然而，这些方法严重依赖于图像级的不一致性，这在解决文本提示的影响方面构成了根本性的限制。在本文中，我们提出了EmotiFlare，这是一种新型的对抗性保护方法，旨在保护图像免受基于扩散的修复模型促进的恶意修改。我们的方法利用交叉注意机制来利用提示嵌入的内在属性。具体来说，我们识别并瞄准不变且语义上无信息的提示共享标记，注入对抗性噪音以抑制采样过程。注入的噪音充当交叉注意诱饵，将模型的焦点从有意义的预算图像对齐上转移，从而抵消提示的影响。在EditBench数据集上进行的大量实验表明，我们的方法在各种指标上实现了最先进的性能，同时显着减少了计算负担和图形处理器内存使用。这些研究结果强调，EmotiFlare是针对未经授权的图像操纵的强大而有效的保护措施。该代码可在https://github.com/NAHOHYUN-SKKU/PromptFlare上获取。



## **43. How to Beat Nakamoto in the Race**

如何在比赛中击败中本聪 cs.CR

Accepted for presentation at the 2025 ACM Conference on Computer and  Communications Security (CCS)

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16202v1) [paper-pdf](http://arxiv.org/pdf/2508.16202v1)

**Authors**: Shu-Jie Cao, Dongning Guo

**Abstract**: This paper studies proof-of-work Nakamoto consensus under bounded network delays, settling two long-standing questions in blockchain security: How can an adversary most effectively attack block safety under a given block confirmation latency? And what is the resulting probability of safety violation? A Markov decision process (MDP) framework is introduced to precise characterize the system state (including the tree and timings of all blocks mined), the adversary's potential actions, and the state transitions due to the adversarial action and the random block arrival processes. An optimal attack, called bait-and-switch, is proposed and proved to maximize the adversary's chance of violating block safety by "beating Nakamoto in the race". The exact probability of this violation is calculated for any confirmation depth using Markov chain analysis, offering fresh insights into the interplay of network delay, confirmation rules, and blockchain security.

摘要: 本文研究了有限网络延迟下的工作量证明中本聪共识，解决了区块链安全中的两个长期存在的问题：在给定的块确认延迟下，对手如何才能最有效地攻击块安全？由此产生的安全违规可能性是多少？引入了马尔科夫决策过程（MDP）框架来精确描述系统状态（包括挖掘的所有块的树和时间）、对手的潜在行为以及由于对抗行为和随机块到达过程而导致的状态转变。提出并证明了一种称为诱饵和开关的最佳攻击，可以通过“在比赛中击败中本聪”来最大化对手违反区块安全的机会。这种违规的确切概率是使用马尔可夫链分析针对任何确认深度计算的，为网络延迟、确认规则和区块链安全性的相互作用提供了新的见解。



## **44. Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks**

评估机器取消学习针对成员推断攻击的防御潜力 cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16150v1) [paper-pdf](http://arxiv.org/pdf/2508.16150v1)

**Authors**: Aristeidis Sidiropoulos, Christos Chrysanthos Nikolaidis, Theodoros Tsiolakis, Nikolaos Pavlidis, Vasilis Perifanis, Pavlos S. Efraimidis

**Abstract**: Membership Inference Attacks (MIAs) pose a significant privacy risk, as they enable adversaries to determine whether a specific data point was included in the training dataset of a model. While Machine Unlearning is primarily designed as a privacy mechanism to efficiently remove private data from a machine learning model without the need for full retraining, its impact on the susceptibility of models to MIA remains an open question. In this study, we systematically assess the vulnerability of models to MIA after applying state-of-art Machine Unlearning algorithms. Our analysis spans four diverse datasets (two from the image domain and two in tabular format), exploring how different unlearning approaches influence the exposure of models to membership inference. The findings highlight that while Machine Unlearning is not inherently a countermeasure against MIA, the unlearning algorithm and data characteristics can significantly affect a model's vulnerability. This work provides essential insights into the interplay between Machine Unlearning and MIAs, offering guidance for the design of privacy-preserving machine learning systems.

摘要: 会员推断攻击（MIA）构成了重大的隐私风险，因为它们使对手能够确定特定数据点是否包含在模型的训练数据集中。虽然Machine Unlearning主要被设计为一种隐私机制，用于有效地从机器学习模型中删除私人数据，而不需要完全重新培训，但它对模型对MIA敏感性的影响仍然是一个悬而未决的问题。在这项研究中，我们在应用最先进的机器取消学习算法后系统评估了模型对MIA的脆弱性。我们的分析跨越了四个不同的数据集（两个来自图像域，两个以表格形式），探索不同的去学习方法如何影响模型对隶属推理的暴露。研究结果强调，虽然机器取消学习本质上并不是针对MIA的对策，但取消学习算法和数据特征可能会显着影响模型的脆弱性。这项工作为机器非学习和MIA之间的相互作用提供了重要见解，为保护隐私的机器学习系统的设计提供了指导。



## **45. Robust Graph Contrastive Learning with Information Restoration**

具有信息恢复的鲁棒图对比学习 cs.LG

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2307.12555v3) [paper-pdf](http://arxiv.org/pdf/2307.12555v3)

**Authors**: Yulin Zhu, Xing Ai, Yevgeniy Vorobeychik, Kai Zhou

**Abstract**: The graph contrastive learning (GCL) framework has gained remarkable achievements in graph representation learning. However, similar to graph neural networks (GNNs), GCL models are susceptible to graph structural attacks. As an unsupervised method, GCL faces greater challenges in defending against adversarial attacks. Furthermore, there has been limited research on enhancing the robustness of GCL. To thoroughly explore the failure of GCL on the poisoned graphs, we investigate the detrimental effects of graph structural attacks against the GCL framework. We discover that, in addition to the conventional observation that graph structural attacks tend to connect dissimilar node pairs, these attacks also diminish the mutual information between the graph and its representations from an information-theoretical perspective, which is the cornerstone of the high-quality node embeddings for GCL. Motivated by this theoretical insight, we propose a robust graph contrastive learning framework with a learnable sanitation view that endeavors to sanitize the augmented graphs by restoring the diminished mutual information caused by the structural attacks. Additionally, we design a fully unsupervised tuning strategy to tune the hyperparameters without accessing the label information, which strictly coincides with the defender's knowledge. Extensive experiments demonstrate the effectiveness and efficiency of our proposed method compared to competitive baselines.

摘要: 图对比学习（GCL）框架在图表示学习方面取得了显着的成就。然而，与图神经网络（GNN）类似，GCL模型也容易受到图结构攻击。作为一种无监督方法，GCL在防御对抗攻击方面面临着更大的挑战。此外，关于增强GCL稳健性的研究有限。为了彻底探讨GCL对中毒图的失败，我们研究了图结构攻击对GCL框架的有害影响。我们发现，除了图结构攻击倾向于连接不同的节点对的传统观察之外，这些攻击还从信息理论的角度减少了图及其表示之间的互信息，这是GCL高质量节点嵌入的基石。受这一理论见解的启发，我们提出了一个强大的图形对比学习框架，具有可学习的卫生观点，该框架致力于通过恢复结构性攻击导致的减少的互信息来净化增强的图形。此外，我们设计了一种完全无监督的调整策略，在不访问标签信息的情况下调整超参数，这与防御者的知识严格一致。大量实验证明了与竞争基线相比，我们提出的方法的有效性和效率。



## **46. Who's the Evil Twin? Differential Auditing for Undesired Behavior**

谁是邪恶双胞胎？针对不良行为的差异审计 cs.LG

main section: 8 pages, 4 figures, 1 table total: 34 pages, 44  figures, 12 tables

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.06827v2) [paper-pdf](http://arxiv.org/pdf/2508.06827v2)

**Authors**: Ishwar Balappanawar, Venkata Hasith Vattikuti, Greta Kintzley, Ronan Azimi-Mancel, Satvik Golechha

**Abstract**: Detecting hidden behaviors in neural networks poses a significant challenge due to minimal prior knowledge and potential adversarial obfuscation. We explore this problem by framing detection as an adversarial game between two teams: the red team trains two similar models, one trained solely on benign data and the other trained on data containing hidden harmful behavior, with the performance of both being nearly indistinguishable on the benign dataset. The blue team, with limited to no information about the harmful behaviour, tries to identify the compromised model. We experiment using CNNs and try various blue team strategies, including Gaussian noise analysis, model diffing, integrated gradients, and adversarial attacks under different levels of hints provided by the red team. Results show high accuracy for adversarial-attack-based methods (100\% correct prediction, using hints), which is very promising, whilst the other techniques yield more varied performance. During our LLM-focused rounds, we find that there are not many parallel methods that we could apply from our study with CNNs. Instead, we find that effective LLM auditing methods require some hints about the undesired distribution, which can then used in standard black-box and open-weight methods to probe the models further and reveal their misalignment. We open-source our auditing games (with the model and data) and hope that our findings contribute to designing better audits.

摘要: 由于先验知识最少和潜在的对抗混淆，检测神经网络中的隐藏行为构成了重大挑战。我们通过将检测视为两个团队之间的对抗游戏来探索这个问题：红团队训练两个类似的模型，一个仅在良性数据上训练，另一个在包含隐藏有害行为的数据上训练，两者的性能在良性数据集上几乎无法区分。蓝色团队在几乎没有有关有害行为的信息的情况下，试图识别受损害的模型。我们使用CNN进行实验，并尝试各种蓝队策略，包括高斯噪音分析、模型差异、综合梯度以及红队提供的不同级别提示下的对抗攻击。结果显示基于对抗攻击的方法具有高准确性（使用提示，100%正确预测），这非常有希望，而其他技术则产生了更多样化的性能。在我们以LLM为重点的回合中，我们发现我们可以从CNN研究中应用的并行方法并不多。相反，我们发现有效的LLM审计方法需要一些有关不希望的分布的提示，然后可以在标准黑匣子和开权方法中使用这些提示，以进一步探索模型并揭示它们的失调。我们开源了审计游戏（包含模型和数据），并希望我们的发现有助于设计更好的审计。



## **47. Distributed Detection of Adversarial Attacks in Multi-Agent Reinforcement Learning with Continuous Action Space**

具有连续动作空间的多智能体强化学习中对抗性攻击的分布式检测 cs.LG

Accepted for publication at ECAI 2025

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15764v1) [paper-pdf](http://arxiv.org/pdf/2508.15764v1)

**Authors**: Kiarash Kazari, Ezzeldin Shereen, György Dán

**Abstract**: We address the problem of detecting adversarial attacks against cooperative multi-agent reinforcement learning with continuous action space. We propose a decentralized detector that relies solely on the local observations of the agents and makes use of a statistical characterization of the normal behavior of observable agents. The proposed detector utilizes deep neural networks to approximate the normal behavior of agents as parametric multivariate Gaussian distributions. Based on the predicted density functions, we define a normality score and provide a characterization of its mean and variance. This characterization allows us to employ a two-sided CUSUM procedure for detecting deviations of the normality score from its mean, serving as a detector of anomalous behavior in real-time. We evaluate our scheme on various multi-agent PettingZoo benchmarks against different state-of-the-art attack methods, and our results demonstrate the effectiveness of our method in detecting impactful adversarial attacks. Particularly, it outperforms the discrete counterpart by achieving AUC-ROC scores of over 0.95 against the most impactful attacks in all evaluated environments.

摘要: 我们解决了检测针对具有连续动作空间的合作多智能体强化学习的对抗攻击的问题。我们提出了一种去中心化的检测器，它仅依赖于代理的局部观察，并利用可观察代理的正常行为的统计特征。提出的检测器利用深度神经网络将代理的正常行为逼近为参数多元高斯分布。基于预测的密度函数，我们定义正态分数并提供其均值和方差的描述。这种特征使我们能够使用双边CLARUM程序来检测正态分数与其平均值的偏差，作为实时异常行为的检测器。我们在各种多代理PettingZoo基准上针对不同最先进的攻击方法评估了我们的方案，我们的结果证明了我们的方法在检测有影响力的对抗性攻击方面的有效性。特别是，它在所有评估环境中针对最具影响力的攻击时，其AUC-ROC得分超过0.95，优于离散对应的。



## **48. Let's Measure Information Step-by-Step: LLM-Based Evaluation Beyond Vibes**

让我们逐步衡量信息：超越共鸣的基于LLM的评估 cs.LG

Add AUC results, pre-reg conformance, theory section clarification.  12 pages

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.05469v2) [paper-pdf](http://arxiv.org/pdf/2508.05469v2)

**Authors**: Zachary Robertson, Sanmi Koyejo

**Abstract**: We study evaluation of AI systems without ground truth by exploiting a link between strategic gaming and information loss. We analyze which information-theoretic mechanisms resist adversarial manipulation, extending finite-sample bounds to show that bounded f-divergences (e.g., total variation distance) maintain polynomial guarantees under attacks while unbounded measures (e.g., KL divergence) degrade exponentially. To implement these mechanisms, we model the overseer as an agent and characterize incentive-compatible scoring rules as f-mutual information objectives. Under adversarial attacks, TVD-MI maintains effectiveness (area under curve 0.70-0.77) while traditional judge queries are near change (AUC $\approx$ 0.50), demonstrating that querying the same LLM for information relationships rather than quality judgments provides both theoretical and practical robustness. The mechanisms decompose pairwise evaluations into reliable item-level quality scores without ground truth, addressing a key limitation of traditional peer prediction. We release preregistration and code.

摘要: 我们通过利用战略游戏和信息丢失之间的联系来研究在没有基本真相的情况下对人工智能系统的评估。我们分析哪些信息论机制抵抗对抗操纵，扩展有限样本界限以表明有界f-分歧（例如，总变化距离）在攻击下保持多项保证，而无界措施（例如，KL分歧）呈指数级下降。为了实现这些机制，我们将监督者建模为代理，并将激励兼容评分规则描述为f-互信息目标。在对抗性攻击下，TVD-MI保持有效性（曲线下面积0.70-0.77），而传统的判断查询几乎发生变化（UC $\大约$0.50），这表明查询相同的LLM以获取信息关系而不是质量判断提供了理论和实践的鲁棒性。这些机制将成对评估分解为可靠的项目级质量分数，无需基本真相，解决了传统同伴预测的关键局限性。我们发布预注册和代码。



## **49. Towards a 3D Transfer-based Black-box Attack via Critical Feature Guidance**

通过关键功能指导进行基于3D传输的黑匣子攻击 cs.CV

11 pages, 6 figures

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15650v1) [paper-pdf](http://arxiv.org/pdf/2508.15650v1)

**Authors**: Shuchao Pang, Zhenghan Chen, Shen Zhang, Liming Lu, Siyuan Liang, Anan Du, Yongbin Zhou

**Abstract**: Deep neural networks for 3D point clouds have been demonstrated to be vulnerable to adversarial examples. Previous 3D adversarial attack methods often exploit certain information about the target models, such as model parameters or outputs, to generate adversarial point clouds. However, in realistic scenarios, it is challenging to obtain any information about the target models under conditions of absolute security. Therefore, we focus on transfer-based attacks, where generating adversarial point clouds does not require any information about the target models. Based on our observation that the critical features used for point cloud classification are consistent across different DNN architectures, we propose CFG, a novel transfer-based black-box attack method that improves the transferability of adversarial point clouds via the proposed Critical Feature Guidance. Specifically, our method regularizes the search of adversarial point clouds by computing the importance of the extracted features, prioritizing the corruption of critical features that are likely to be adopted by diverse architectures. Further, we explicitly constrain the maximum deviation extent of the generated adversarial point clouds in the loss function to ensure their imperceptibility. Extensive experiments conducted on the ModelNet40 and ScanObjectNN benchmark datasets demonstrate that the proposed CFG outperforms the state-of-the-art attack methods by a large margin.

摘要: 用于3D点云的深度神经网络已被证明容易受到对抗性示例的影响。之前的3D对抗攻击方法通常利用有关目标模型的某些信息（例如模型参数或输出）来生成对抗点云。然而，在现实场景中，在绝对安全的条件下获取有关目标模型的任何信息是具有挑战性的。因此，我们专注于基于传输的攻击，其中生成对抗点云不需要有关目标模型的任何信息。根据我们观察到用于点云分类的关键特征在不同的DNN架构中是一致的，我们提出了CGM，这是一种新型的基于传输的黑匣子攻击方法，通过提出的关键特征指南提高了对抗性点云的可传输性。具体来说，我们的方法通过计算提取特征的重要性、优先考虑可能被不同架构采用的关键特征的损坏，来规范对抗性点云的搜索。此外，我们在损失函数中明确限制生成的对抗点云的最大偏差程度，以确保其不可感知性。在Model Net 40和ScanObjectNN基准数据集上进行的大量实验表明，拟议的CGM的性能大大优于最先进的攻击方法。



## **50. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

人工智能生成图像检测中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2407.20836v4) [paper-pdf](http://arxiv.org/pdf/2407.20836v4)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.

摘要: 图像合成领域的最新进展，特别是GAN和扩散模型的出现，加剧了公众对虚假信息传播的担忧。为了解决这些问题，人们提出了许多人工智能生成的图像（AIGI）检测器，并在识别虚假图像方面取得了良好的性能。然而，人们仍然缺乏对AIGI检测器对抗鲁棒性的系统了解。在本文中，我们研究了最先进的AIGI检测器在白盒和黑盒设置下对抗攻击的脆弱性，迄今为止，这很少被研究。为此，我们提出了一种攻击AIGI检测器的新方法。首先，受到频域中真实图像和假图像之间明显差异的启发，我们在频域下添加扰动，以推动图像远离其原始频率分布。其次，我们探索代理模型的完整后验分布，以进一步缩小异类AIGI检测器之间的差距，例如跨CNN和ViT传输对抗性示例。这是通过引入一种新型的训练后Bayesian策略来实现的，该策略将单个代理变成了Bayesian代理，能够使用一个预先训练的代理来模拟不同的受害者模型，而无需重新训练。我们将我们的方法命名为基于频率的训练后Bayesian Attack（FPBA）。通过FPBA，我们表明对抗性攻击确实是对AIGI检测器的真正威胁，因为FPBA可以跨模型、生成器、防御方法提供成功的黑匣子攻击，甚至逃避交叉生成器检测，这是一个至关重要的现实世界检测场景。该代码将在接受后共享。



