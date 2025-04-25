# Latest Adversarial Attack Papers
**update at 2025-04-25 10:06:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Siren -- Advancing Cybersecurity through Deception and Adaptive Analysis**

警报器--通过欺骗和适应性分析推进网络安全 cs.CR

14 pages, 5 figures, 13th Computing Conference 2025 - London, United  Kingdom

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2406.06225v2) [paper-pdf](http://arxiv.org/pdf/2406.06225v2)

**Authors**: Samhruth Ananthanarayanan, Girish Kulathumani, Ganesh Narayanan

**Abstract**: Siren represents a pioneering research effort aimed at fortifying cybersecurity through strategic integration of deception, machine learning, and proactive threat analysis. Drawing inspiration from mythical sirens, this project employs sophisticated methods to lure potential threats into controlled environments. The system features a dynamic machine learning model for realtime analysis and classification, ensuring continuous adaptability to emerging cyber threats. The architectural framework includes a link monitoring proxy, a purpose-built machine learning model for dynamic link analysis, and a honeypot enriched with simulated user interactions to intensify threat engagement. Data protection within the honeypot is fortified with probabilistic encryption. Additionally, the incorporation of simulated user activity extends the system's capacity to capture and learn from potential attackers even after user disengagement. Overall, Siren introduces a paradigm shift in cybersecurity, transforming traditional defense mechanisms into proactive systems that actively engage and learn from potential adversaries. The research strives to enhance user protection while yielding valuable insights for ongoing refinement in response to the evolving landscape of cybersecurity threats.

摘要: Siren代表了一项开创性的研究工作，旨在通过欺骗、机器学习和主动威胁分析的战略集成来加强网络安全。该项目从神话警报中汲取灵感，采用复杂的方法将潜在威胁引诱到受控环境中。该系统具有动态机器学习模型，用于实时分析和分类，确保对新兴网络威胁的持续适应性。该架构框架包括一个链接监控代理、一个专门构建的用于动态链接分析的机器学习模型，以及一个富含模拟用户交互以加强威胁参与的蜜罐。蜜罐内的数据保护通过概率加密得到加强。此外，模拟用户活动的结合扩展了系统捕获潜在攻击者并从其学习的能力，即使在用户脱离接触之后也是如此。总体而言，Siren引入了网络安全的范式转变，将传统防御机制转变为积极参与潜在对手并向其学习的主动系统。该研究致力于加强用户保护，同时为持续改进提供有价值的见解，以应对不断变化的网络安全威胁格局。



## **2. On the Generalization of Adversarially Trained Quantum Classifiers**

关于对抗训练量子分类器的推广 quant-ph

22 pages, 6 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17690v1) [paper-pdf](http://arxiv.org/pdf/2504.17690v1)

**Authors**: Petros Georgiou, Aaron Mark Thomas, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: Quantum classifiers are vulnerable to adversarial attacks that manipulate their input classical or quantum data. A promising countermeasure is adversarial training, where quantum classifiers are trained by using an attack-aware, adversarial loss function. This work establishes novel bounds on the generalization error of adversarially trained quantum classifiers when tested in the presence of perturbation-constrained adversaries. The bounds quantify the excess generalization error incurred to ensure robustness to adversarial attacks as scaling with the training sample size $m$ as $1/\sqrt{m}$, while yielding insights into the impact of the quantum embedding. For quantum binary classifiers employing \textit{rotation embedding}, we find that, in the presence of adversarial attacks on classical inputs $\mathbf{x}$, the increase in sample complexity due to adversarial training over conventional training vanishes in the limit of high dimensional inputs $\mathbf{x}$. In contrast, when the adversary can directly attack the quantum state $\rho(\mathbf{x})$ encoding the input $\mathbf{x}$, the excess generalization error depends on the choice of embedding only through its Hilbert space dimension. The results are also extended to multi-class classifiers. We validate our theoretical findings with numerical experiments.

摘要: 量子分类器容易受到操纵其输入经典或量子数据的对抗性攻击。一个有希望的对策是对抗训练，其中量子分类器通过使用攻击感知的对抗损失函数来训练。这项工作建立了新的边界对抗训练的量子分类器的泛化错误时，在扰动约束的对手的存在下进行测试。边界量化了为确保对抗性攻击的鲁棒性而产生的过度泛化误差，并将训练样本大小$m$缩放为$1/\sqrt{m}$，同时深入了解量子嵌入的影响。对于采用旋转嵌入的量子二进制分类器，我们发现，在经典输入$\mathbf{x}$上存在对抗性攻击的情况下，由于对抗性训练而导致的样本复杂度增加在高维输入$\mathbf{x}$的限制下消失。相比之下，当对手可以直接攻击量子态$\rho（\mathbf{x}）$编码输入$\mathbf{x}$时，多余的泛化误差取决于仅通过其希尔伯特空间维度嵌入的选择。结果也被扩展到多类分类器。我们验证了我们的理论研究结果与数值实验。



## **3. Evaluating the Vulnerability of ML-Based Ethereum Phishing Detectors to Single-Feature Adversarial Perturbations**

评估基于ML的以太坊网络钓鱼检测器对单一特征对抗扰动的脆弱性 cs.CR

24 pages; an extension of a paper that appeared at WISA 2024

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17684v1) [paper-pdf](http://arxiv.org/pdf/2504.17684v1)

**Authors**: Ahod Alghuried, Ali Alkinoon, Abdulaziz Alghamdi, Soohyeon Choi, Manar Mohaisen, David Mohaisen

**Abstract**: This paper explores the vulnerability of machine learning models to simple single-feature adversarial attacks in the context of Ethereum fraudulent transaction detection. Through comprehensive experimentation, we investigate the impact of various adversarial attack strategies on model performance metrics. Our findings, highlighting how prone those techniques are to simple attacks, are alarming, and the inconsistency in the attacks' effect on different algorithms promises ways for attack mitigation. We examine the effectiveness of different mitigation strategies, including adversarial training and enhanced feature selection, in enhancing model robustness and show their effectiveness.

摘要: 本文探讨了以太坊欺诈交易检测背景下机器学习模型对简单单特征对抗攻击的脆弱性。通过全面的实验，我们研究了各种对抗攻击策略对模型性能指标的影响。我们的研究结果强调了这些技术容易受到简单攻击的影响，这令人震惊，而且攻击对不同算法的影响的不一致性为缓解攻击提供了方法。我们研究了不同缓解策略（包括对抗训练和增强的特征选择）在增强模型稳健性方面的有效性，并展示了它们的有效性。



## **4. Regulatory Markets for AI Safety**

人工智能安全监管市场 cs.CY

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2001.00078v2) [paper-pdf](http://arxiv.org/pdf/2001.00078v2)

**Authors**: Jack Clark, Gillian K. Hadfield

**Abstract**: We propose a new model for regulation to achieve AI safety: global regulatory markets. We first sketch the model in general terms and provide an overview of the costs and benefits of this approach. We then demonstrate how the model might work in practice: responding to the risk of adversarial attacks on AI models employed in commercial drones.

摘要: 我们提出了一种实现人工智能安全的新监管模式：全球监管市场。我们首先概述该模型，并概述这种方法的成本和收益。然后，我们演示了该模型在实践中如何工作：应对商用无人机中使用的人工智能模型的对抗攻击风险。



## **5. GRANITE : a Byzantine-Resilient Dynamic Gossip Learning Framework**

GRANITE：一个具有拜占庭弹性的动态八卦学习框架 cs.LG

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17471v1) [paper-pdf](http://arxiv.org/pdf/2504.17471v1)

**Authors**: Yacine Belal, Mohamed Maouche, Sonia Ben Mokhtar, Anthony Simonet-Boulogne

**Abstract**: Gossip Learning (GL) is a decentralized learning paradigm where users iteratively exchange and aggregate models with a small set of neighboring peers. Recent GL approaches rely on dynamic communication graphs built and maintained using Random Peer Sampling (RPS) protocols. Thanks to graph dynamics, GL can achieve fast convergence even over extremely sparse topologies. However, the robustness of GL over dy- namic graphs to Byzantine (model poisoning) attacks remains unaddressed especially when Byzantine nodes attack the RPS protocol to scale up model poisoning. We address this issue by introducing GRANITE, a framework for robust learning over sparse, dynamic graphs in the presence of a fraction of Byzantine nodes. GRANITE relies on two key components (i) a History-aware Byzantine-resilient Peer Sampling protocol (HaPS), which tracks previously encountered identifiers to reduce adversarial influence over time, and (ii) an Adaptive Probabilistic Threshold (APT), which leverages an estimate of Byzantine presence to set aggregation thresholds with formal guarantees. Empirical results confirm that GRANITE maintains convergence with up to 30% Byzantine nodes, improves learning speed via adaptive filtering of poisoned models and obtains these results in up to 9 times sparser graphs than dictated by current theory.

摘要: Gossip Learning（GL）是一种去中心化的学习范式，用户与一小群邻近的对等点迭代地交换和聚合模型。最近的GL方法依赖于使用随机对等采样（RPS）协议构建和维护的动态通信图。得益于图形动态学，GL即使在极其稀疏的拓扑上也可以实现快速收敛。然而，GL在动态图上对拜占庭（模型中毒）攻击的鲁棒性仍然没有得到解决，尤其是当拜占庭节点攻击RPS协议以扩大模型中毒规模时。我们通过引入GRANITE来解决这个问题，GRANITE是一个在存在一小部分拜占庭节点的情况下对稀疏动态图进行鲁棒学习的框架。GRANITE依赖于两个关键组件（i）历史感知拜占庭弹性对等采样协议（HaPS），它跟踪之前遇到的标识符，以随着时间的推移减少对抗影响，以及（ii）自适应概率阈值（APT），它利用拜占庭存在的估计来设置具有正式保证的聚合阈值。经验结果证实，GRANITE可以在高达30%的拜占庭节点上保持收敛，通过有毒模型的自适应过滤提高学习速度，并在比当前理论规定的稀疏9倍的图中获得这些结果。



## **6. Evaluating Time Series Models for Urban Wastewater Management: Predictive Performance, Model Complexity and Resilience**

评估城市废水管理的时间序列模型：预测性能、模型复杂性和弹性 cs.LG

6 pages, 6 figures, accepted at 10th International Conference on  Smart and Sustainable Technologies (SpliTech) 2025, GitHub:  https://github.com/calgo-lab/resilient-timeseries-evaluation

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17461v1) [paper-pdf](http://arxiv.org/pdf/2504.17461v1)

**Authors**: Vipin Singh, Tianheng Ling, Teodor Chiaburu, Felix Biessmann

**Abstract**: Climate change increases the frequency of extreme rainfall, placing a significant strain on urban infrastructures, especially Combined Sewer Systems (CSS). Overflows from overburdened CSS release untreated wastewater into surface waters, posing environmental and public health risks. Although traditional physics-based models are effective, they are costly to maintain and difficult to adapt to evolving system dynamics. Machine Learning (ML) approaches offer cost-efficient alternatives with greater adaptability. To systematically assess the potential of ML for modeling urban infrastructure systems, we propose a protocol for evaluating Neural Network architectures for CSS time series forecasting with respect to predictive performance, model complexity, and robustness to perturbations. In addition, we assess model performance on peak events and critical fluctuations, as these are the key regimes for urban wastewater management. To investigate the feasibility of lightweight models suitable for IoT deployment, we compare global models, which have access to all information, with local models, which rely solely on nearby sensor readings. Additionally, to explore the security risks posed by network outages or adversarial attacks on urban infrastructure, we introduce error models that assess the resilience of models. Our results demonstrate that while global models achieve higher predictive performance, local models provide sufficient resilience in decentralized scenarios, ensuring robust modeling of urban infrastructure. Furthermore, models with longer native forecast horizons exhibit greater robustness to data perturbations. These findings contribute to the development of interpretable and reliable ML solutions for sustainable urban wastewater management. The implementation is available in our GitHub repository.

摘要: 气候变化增加了极端降雨的频率，给城市基础设施，尤其是联合下水道系统（CSS）带来了巨大压力。不堪重负的CSS溢流将未经处理的废水排放到地表水中，构成环境和公共健康风险。尽管传统的基于物理的模型是有效的，但它们的维护成本高昂，并且难以适应不断变化的系统动态。机器学习（ML）方法提供了具有更强适应性的经济高效替代方案。为了系统地评估ML对城市基础设施系统建模的潜力，我们提出了一种用于评估CSS时间序列预测的神经网络架构的协议，包括预测性能、模型复杂性和对扰动的鲁棒性。此外，我们还评估了峰值事件和临界波动的模型性能，因为这些是城市废水管理的关键制度。为了研究适合物联网部署的轻量级模型的可行性，我们将可以访问所有信息的全球模型与仅依赖附近传感器读数的本地模型进行了比较。此外，为了探索网络中断或对城市基础设施的对抗性攻击带来的安全风险，我们引入了评估模型弹性的错误模型。我们的结果表明，虽然全球模型实现了更高的预测性能，但本地模型在去中心化场景中提供了足够的弹性，确保了城市基础设施的稳健建模。此外，具有更长本地预测视野的模型对数据扰动表现出更强的鲁棒性。这些发现有助于开发可解释且可靠的ML解决方案，用于可持续城市废水管理。该实现可在我们的GitHub存储库中提供。



## **7. Unveiling Hidden Vulnerabilities in Digital Human Generation via Adversarial Attacks**

通过对抗性攻击揭露数字人类世代中隐藏的漏洞 cs.CV

14 pages, 7 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17457v1) [paper-pdf](http://arxiv.org/pdf/2504.17457v1)

**Authors**: Zhiying Li, Yeying Jin, Fan Shen, Zhi Liu, Weibin Chen, Pengju Zhang, Xiaomei Zhang, Boyu Chen, Michael Shen, Kejian Wu, Zhaoxin Fan, Jin Dong

**Abstract**: Expressive human pose and shape estimation (EHPS) is crucial for digital human generation, especially in applications like live streaming. While existing research primarily focuses on reducing estimation errors, it largely neglects robustness and security aspects, leaving these systems vulnerable to adversarial attacks. To address this significant challenge, we propose the \textbf{Tangible Attack (TBA)}, a novel framework designed to generate adversarial examples capable of effectively compromising any digital human generation model. Our approach introduces a \textbf{Dual Heterogeneous Noise Generator (DHNG)}, which leverages Variational Autoencoders (VAE) and ControlNet to produce diverse, targeted noise tailored to the original image features. Additionally, we design a custom \textbf{adversarial loss function} to optimize the noise, ensuring both high controllability and potent disruption. By iteratively refining the adversarial sample through multi-gradient signals from both the noise and the state-of-the-art EHPS model, TBA substantially improves the effectiveness of adversarial attacks. Extensive experiments demonstrate TBA's superiority, achieving a remarkable 41.0\% increase in estimation error, with an average improvement of approximately 17.0\%. These findings expose significant security vulnerabilities in current EHPS models and highlight the need for stronger defenses in digital human generation systems.

摘要: 表现性人体姿势和形状估计（EHPS）对于数字人类生成至关重要，尤其是在直播等应用中。虽然现有的研究主要关注于减少估计误差，但它在很大程度上忽视了稳健性和安全性方面，使这些系统容易受到对抗性攻击。为了解决这一重大挑战，我们提出了\textBF{TSYS Attack（TBA）}，这是一个新颖的框架，旨在生成能够有效损害任何数字人类生成模型的对抗性示例。我们的方法引入了\textBF{Dual Heterogeneous Noise Generator（DHNG）}，它利用变分自动编码器（VAE）和Control Net来产生针对原始图像特征定制的多样化、有针对性的噪音。此外，我们还设计了一个自定义的\textBF{对抗损失函数}来优化噪音，确保高可控性和强干扰。通过通过来自噪音和最先进的EHPS模型的多梯度信号迭代细化对抗样本，TBA大幅提高了对抗攻击的有效性。大量实验证明了TBA的优越性，估计误差显着增加了41.0%，平均改进约为17.0%。这些发现暴露了当前EHPS模型中的重大安全漏洞，并凸显了数字人类生成系统中需要更强的防御。



## **8. Analysis and Mitigation of Data injection Attacks against Data-Driven Control**

针对数据驱动控制的数据注入攻击的分析和缓解 eess.SY

Under review for publication

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17347v1) [paper-pdf](http://arxiv.org/pdf/2504.17347v1)

**Authors**: Sribalaji C. Anand

**Abstract**: This paper investigates the impact of false data injection attacks on data-driven control systems. Specifically, we consider an adversary injecting false data into the sensor channels during the learning phase. When the operator seeks to learn a stable state-feedback controller, we propose an attack strategy capable of misleading the operator into learning an unstable feedback gain. We also investigate the effects of constant-bias injection attacks on data-driven linear quadratic regulation (LQR). Finally, we explore potential mitigation strategies and support our findings with numerical examples.

摘要: 本文研究了虚假数据注入攻击对数据驱动控制系统的影响。具体来说，我们考虑对手在学习阶段将错误数据注入传感器通道。当操作员寻求学习稳定的状态反馈控制器时，我们提出了一种能够误导操作员学习不稳定的反馈收益的攻击策略。我们还研究了恒定偏差注入攻击对数据驱动线性二次调节（LQR）的影响。最后，我们探索潜在的缓解策略，并通过数字示例支持我们的发现。



## **9. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

越狱镜头：以表象和电路的视角解读越狱机制 cs.CR

17 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2411.11114v2) [paper-pdf](http://arxiv.org/pdf/2411.11114v2)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Wenhui Zhang, Qinglong Wang, Rui Zheng

**Abstract**: Despite the outstanding performance of Large language Models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses. Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explained typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of jailbreak attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives~(which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on five mainstream LLMs under seven jailbreak strategies. Our evaluation reveals that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. This manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals. Notably, we find a strong and consistent correlation between representation deception and activation shift of key circuits across diverse jailbreak methods and multiple LLMs.

摘要: 尽管大型语言模型（LLM）在不同任务中表现出色，但它们很容易受到越狱攻击，其中对抗性提示是为了绕过其安全机制并引发意外响应。尽管越狱攻击很普遍，但对其潜在机制的了解仍然有限。最近的研究解释了典型的越狱行为（例如，模型拒绝响应的程度）通过分析越狱提示引起的LLM潜在空间的表示变化或识别有助于越狱攻击成功的关键神经元。然而，这些研究既没有探索不同的越狱模式，也没有提供从电路故障到表象变化的细粒度解释，在揭示越狱机制方面留下了重大空白。在本文中，我们提出了JailbreakLens，这是一个解释框架，从表示（揭示了越狱如何改变模型的危害性感知）和电路视角（通过识别导致漏洞的关键电路来揭示这些欺骗的原因）来分析越狱机制），在整个响应生成过程中跟踪它们的演变。然后，我们对七种越狱策略下的五种主流LLM的越狱行为进行了深入评估。我们的评估表明，越狱会放大强化肯定反应的成分，同时抑制那些产生拒绝反应的成分。这种操纵将模型表示转移到安全的集群以欺骗LLM，导致其提供详细的响应而不是拒绝。值得注意的是，我们发现不同越狱方法和多种LLM中的代表欺骗和关键电路的激活转变之间存在强烈且一致的相关性。



## **10. Enhancing Variational Autoencoders with Smooth Robust Latent Encoding**

通过平滑稳健的隐性编码增强变分自动编码器 cs.LG

Under review

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17219v1) [paper-pdf](http://arxiv.org/pdf/2504.17219v1)

**Authors**: Hyomin Lee, Minseon Kim, Sangwon Jang, Jongheon Jeong, Sung Ju Hwang

**Abstract**: Variational Autoencoders (VAEs) have played a key role in scaling up diffusion-based generative models, as in Stable Diffusion, yet questions regarding their robustness remain largely underexplored. Although adversarial training has been an established technique for enhancing robustness in predictive models, it has been overlooked for generative models due to concerns about potential fidelity degradation by the nature of trade-offs between performance and robustness. In this work, we challenge this presumption, introducing Smooth Robust Latent VAE (SRL-VAE), a novel adversarial training framework that boosts both generation quality and robustness. In contrast to conventional adversarial training, which focuses on robustness only, our approach smooths the latent space via adversarial perturbations, promoting more generalizable representations while regularizing with originality representation to sustain original fidelity. Applied as a post-training step on pre-trained VAEs, SRL-VAE improves image robustness and fidelity with minimal computational overhead. Experiments show that SRL-VAE improves both generation quality, in image reconstruction and text-guided image editing, and robustness, against Nightshade attacks and image editing attacks. These results establish a new paradigm, showing that adversarial training, once thought to be detrimental to generative models, can instead enhance both fidelity and robustness.

摘要: 变分自动编码器（VAE）在扩展基于扩散的生成模型方面发挥了关键作用，如稳定扩散，但有关其鲁棒性的问题仍然在很大程度上未被探索。虽然对抗性训练已经是一种用于增强预测模型鲁棒性的成熟技术，但由于担心性能和鲁棒性之间的权衡可能会导致保真度下降，因此它一直被忽视。在这项工作中，我们挑战了这一假设，引入了平滑鲁棒潜在VAE（SRL-VAE），这是一种新型的对抗训练框架，可以提高生成质量和鲁棒性。与只关注鲁棒性的传统对抗性训练相比，我们的方法通过对抗性扰动来平滑潜在空间，促进更可概括的表示，同时用原始表示进行正则化以维持原始保真度。SRL-VAE作为预训练VAE的训练后步骤应用，以最小的计算负担提高了图像的稳健性和保真度。实验表明，SRL-VAE提高了图像重建和文本引导图像编辑的生成质量，以及针对Nightshade攻击和图像编辑攻击的鲁棒性。这些结果建立了一个新的范式，表明曾经被认为对生成模型有害的对抗训练可以增强保真度和稳健性。



## **11. CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent**

CheatAgent：通过LLM代理攻击LLM授权的推荐系统 cs.CR

Accepted by KDD 2024;

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.13192v2) [paper-pdf](http://arxiv.org/pdf/2504.13192v2)

**Authors**: Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, Feiran Huang

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.

摘要: 最近，基于大语言模型（LLM）的推荐系统（RecSys）在个性化用户体验方面带来了显着进步，并引起了相当大的关注。尽管取得了令人印象深刻的进展，但有关LLM授权的RecSys安全漏洞的研究问题仍然基本上没有得到充分的调查。考虑到安全和隐私问题，更实际的做法是专注于攻击黑匣子RecSys，攻击者只能观察系统的输入和输出。然而，由于处理复杂文本输入、规划和推理的能力有限，使用强化学习（RL）代理的传统攻击方法对于攻击LLM授权的RecSys并不有效。另一方面，LLM提供了前所未有的机会作为攻击代理来攻击RecSys，因为它们在模拟类人决策过程方面具有令人印象深刻的能力。因此，在本文中，我们通过利用LLM的类人能力提出了一种名为CheatAgent的新型攻击框架，其中开发了一个基于LLM的代理来攻击LLM授权的RecSys。具体来说，我们的方法首先识别插入位置，以最小的输入修改获得最大影响。之后，LLM代理被设计为生成对抗性扰动以插入目标位置。为了进一步提高生成的扰动的质量，我们利用即时调整技术通过受害者RecSys的反馈迭代改进攻击策略。三个现实世界数据集的广泛实验证明了我们提出的攻击方法的有效性。



## **12. Range and Topology Mutation Based Wireless Agility**

基于范围和布局突变的无线敏捷性 cs.ET

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17164v1) [paper-pdf](http://arxiv.org/pdf/2504.17164v1)

**Authors**: Qi Duan, Ehab Al-Shae, Jiang Xie

**Abstract**: In this paper, we present formal foundations for two wireless agility techniques: (1) Random Range Mutation (RNM) that allows for periodic changes of AP coverage range randomly, and (2) Ran- dom Topology Mutation (RTM) that allows for random motion and placement of APs in the wireless infrastructure. The goal of these techniques is to proactively defend against targeted attacks (e.g., DoS and eavesdropping) by forcing the wireless clients to change their AP association randomly. We apply Satisfiability Modulo The- ories (SMT) and Answer Set Programming (ASP) based constraint solving methods that allow for optimizing wireless AP mutation while maintaining service requirements including coverage, secu- rity and energy properties under incomplete information about the adversary strategies. Our evaluation validates the feasibility, scalability, and effectiveness of the formal methods based technical approaches.

摘要: 在本文中，我们介绍了两种无线敏捷技术的正式基础：（1）随机范围突变（RNI），允许AP覆盖范围随机定期改变，以及（2）随机Topology Mutation（STM），允许AP在无线基础设施中的随机移动和放置。这些技术的目标是主动防御针对性攻击（例如，通过强制无线客户端随机更改其AP关联来进行拒绝服务和窃听。我们应用基于可满足性模（SMT）和答案集编程（ISP）的约束求解方法，可以优化无线AP突变，同时在有关对手策略的不完整信息下维持包括覆盖范围、安全性和能源属性在内的服务要求。我们的评估验证了基于正式方法的技术方法的可行性、可扩展性和有效性。



## **13. On Minimizing Adversarial Counterfactual Error in Adversarial RL**

对抗性RL中对抗性反事实错误的最小化 cs.LG

Presented at ICLR 2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2406.04724v4) [paper-pdf](http://arxiv.org/pdf/2406.04724v4)

**Authors**: Roman Belaire, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Deep Reinforcement Learning (DRL) policies are highly susceptible to adversarial noise in observations, which poses significant risks in safety-critical scenarios. The challenge inherent to adversarial perturbations is that by altering the information observed by the agent, the state becomes only partially observable. Existing approaches address this by either enforcing consistent actions across nearby states or maximizing the worst-case value within adversarially perturbed observations. However, the former suffers from performance degradation when attacks succeed, while the latter tends to be overly conservative, leading to suboptimal performance in benign settings. We hypothesize that these limitations stem from their failing to account for partial observability directly. To this end, we introduce a novel objective called Adversarial Counterfactual Error (ACoE), defined on the beliefs about the true state and balancing value optimization with robustness. To make ACoE scalable in model-free settings, we propose the theoretically-grounded surrogate objective Cumulative-ACoE (C-ACoE). Our empirical evaluations on standard benchmarks (MuJoCo, Atari, and Highway) demonstrate that our method significantly outperforms current state-of-the-art approaches for addressing adversarial RL challenges, offering a promising direction for improving robustness in DRL under adversarial conditions. Our code is available at https://github.com/romanbelaire/acoe-robust-rl.

摘要: 深度强化学习（DRL）策略极易受到观察中的对抗性噪音的影响，这在安全关键场景中构成了重大风险。对抗性扰动固有的挑战是，通过改变主体观察到的信息，状态变得只能部分可观察。现有的方法要么通过在邻近州之间强制执行一致的行动，要么在敌对扰动的观察中最坏情况的值最大化来解决这个问题。然而，前者在攻击成功时性能会下降，而后者往往过于保守，导致在良性环境下性能次优。我们假设这些限制源于它们未能直接解释部分可观察性。为此，我们引入了一个名为对抗反事实错误（ACOE）的新颖目标，该目标基于对真实状态的信念以及平衡价值优化与鲁棒性。为了使ACOE在无模型设置中可扩展，我们提出了基于理论的代理目标累积ACOE（C-ACOE）。我们对标准基准（MuJoCo、Atari和Highway）的实证评估表明，我们的方法在解决对抗性RL挑战方面显着优于当前最先进的方法，为提高对抗性条件下DRL的稳健性提供了一个有希望的方向。我们的代码可在https://github.com/romanbelaire/acoe-robust-rl上获取。



## **14. BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation**

BadVideo：针对文本转视频生成的秘密后门攻击 cs.CV

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16907v1) [paper-pdf](http://arxiv.org/pdf/2504.16907v1)

**Authors**: Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, Baoyuan Wu

**Abstract**: Text-to-video (T2V) generative models have rapidly advanced and found widespread applications across fields like entertainment, education, and marketing. However, the adversarial vulnerabilities of these models remain rarely explored. We observe that in T2V generation tasks, the generated videos often contain substantial redundant information not explicitly specified in the text prompts, such as environmental elements, secondary objects, and additional details, providing opportunities for malicious attackers to embed hidden harmful content. Exploiting this inherent redundancy, we introduce BadVideo, the first backdoor attack framework tailored for T2V generation. Our attack focuses on designing target adversarial outputs through two key strategies: (1) Spatio-Temporal Composition, which combines different spatiotemporal features to encode malicious information; (2) Dynamic Element Transformation, which introduces transformations in redundant elements over time to convey malicious information. Based on these strategies, the attacker's malicious target seamlessly integrates with the user's textual instructions, providing high stealthiness. Moreover, by exploiting the temporal dimension of videos, our attack successfully evades traditional content moderation systems that primarily analyze spatial information within individual frames. Extensive experiments demonstrate that BadVideo achieves high attack success rates while preserving original semantics and maintaining excellent performance on clean inputs. Overall, our work reveals the adversarial vulnerability of T2V models, calling attention to potential risks and misuse. Our project page is at https://wrt2000.github.io/BadVideo2025/.

摘要: 文本转视频（T2 V）生成模型迅速发展，并在娱乐、教育和营销等领域获得了广泛应用。然而，这些模型的对抗漏洞仍然很少被探讨。我们观察到，在T2 V生成任务中，生成的视频通常包含文本提示中未明确指定的大量冗余信息，例如环境元素、次要对象和其他详细信息，为恶意攻击者提供了嵌入隐藏有害内容的机会。利用这种固有的冗余，我们引入了BadVideo，这是第一个为T2 V一代量身定制的后门攻击框架。我们的攻击重点是通过两个关键策略设计目标对抗输出：（1）时空合成，结合不同的时空特征来编码恶意信息;（2）动态元素转换，引入冗余元素随着时间的推移的转换以传达恶意信息。基于这些策略，攻击者的恶意目标与用户的文本指令无缝集成，提供高度的隐蔽性。此外，通过利用视频的时间维度，我们的攻击成功地规避了主要分析单个帧内空间信息的传统内容审核系统。大量实验表明，BadVideo在保留原始语义并在干净输入上保持出色的性能的同时实现了很高的攻击成功率。总体而言，我们的工作揭示了T2 V模型的对抗脆弱性，并引起人们对潜在风险和滥用的关注。我们的项目页面位于https://wrt2000.github.io/BadVideo2025/。



## **15. aiXamine: Simplified LLM Safety and Security**

aiXamine：简化的LLM安全和安保 cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14985v2) [paper-pdf](http://arxiv.org/pdf/2504.14985v2)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.

摘要: 评估大型语言模型（LLM）的安全性和保障性仍然是一项复杂的任务，通常需要用户在临时基准、数据集、指标和报告格式的碎片化环境中进行导航。为了应对这一挑战，我们推出了aiXamine，这是一个针对LLM安全性的全面黑匣子评估平台。aiXamine集成了40多个测试（即，基准）组织成八个关键服务，针对安全和保障的特定维度：对抗稳健性、代码安全性、公平性和偏见、幻觉、模型和数据隐私、分发外（OOD）稳健性、过度拒绝和安全对齐。该平台将评估结果汇总到每个模型的单个详细报告中，提供模型性能、测试示例和丰富的可视化的详细细分。我们使用aiXamine评估了50多个公开和专有的LLM，进行了超过2000次检查。我们的研究结果揭示了领先模型中的显着漏洞，包括OpenAI GPT-4 o中容易受到对抗攻击、xAI Grok-3中的偏见输出以及Google Gemini 2.0中的隐私弱点。此外，我们观察到开源模型可以在特定服务中匹配或超过专有模型，例如安全性一致、公平性和偏差以及OOD稳健性。最后，我们确定了蒸馏策略、模型大小、训练方法和架构选择之间的权衡。



## **16. A robust and composable device-independent protocol for oblivious transfer using (fully) untrusted quantum devices in the bounded storage model**

在有界存储模型中使用（完全）不受信任的量子设备的稳健且可组合的设备无关协议，用于无意识传输 quant-ph

Major improvement in the main result (security against non-IID  devices)

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2404.11283v2) [paper-pdf](http://arxiv.org/pdf/2404.11283v2)

**Authors**: Rishabh Batra, Sayantan Chakraborty, Rahul Jain, Upendra Kapshikar

**Abstract**: We present a robust and composable device-independent (DI) quantum protocol between two parties for oblivious transfer (OT) using Magic Square devices in the bounded storage model in which the (honest and cheating) devices and parties have no long-term quantum memory. After a fixed constant (real-world) time interval, referred to as DELAY, the quantum states decohere completely. The adversary (cheating party), with full control over the devices, is allowed joint (non-IID) quantum operations on the devices, and there are no time and space complexity bounds placed on its powers. The running time of the honest parties is polylog({\lambda}) (where {\lambda} is the security parameter). Our protocol has negligible (in {\lambda}) correctness and security errors and can be implemented in the NISQ (Noisy Intermediate Scale Quantum) era. By robustness, we mean that our protocol is correct even when devices are slightly off (by a small constant) from their ideal specification. This is an important property since small manufacturing errors in the real-world devices are inevitable. Our protocol is sequentially composable and, hence, can be used as a building block to construct larger protocols (including DI bit-commitment and DI secure multi-party computation) while still preserving correctness and security guarantees.   None of the known DI protocols for OT in the literature are robust and secure against joint quantum attacks. This was a major open question in device-independent two-party distrustful cryptography, which we resolve.   We prove a parallel repetition theorem for a certain class of entangled games with a hybrid (quantum-classical) strategy to show the security of our protocol. The hybrid strategy helps to incorporate DELAY in our protocol. This parallel repetition theorem is a main technical contribution of our work.

摘要: 我们在有界存储模型中使用Magic Square设备，在双方之间提出了一种鲁棒且可组合的设备无关（DI）量子协议，在该模型中，（诚实和作弊的）设备和各方没有长期量子记忆。经过固定的恒定（现实世界）时间间隔（称为延迟）后，量子状态完全去散。对手（作弊方）完全控制设备，允许对设备进行联合（非IID）量子操作，并且对其权力没有时间和空间复杂性限制。诚实各方的运行时间是Polylog（{\ambda}）（其中{\ambda}是安全参数）。我们的协议具有可忽略的（在{\lambda}）的正确性和安全性错误，并可以在NISQ（噪声中间尺度量子）时代实现。所谓鲁棒性，我们的意思是，即使设备稍微偏离其理想规格（通过一个小常数），我们的协议也是正确的。这是一个重要的属性，因为在现实世界的设备中，小的制造误差是不可避免的。我们的协议是顺序可组合的，因此，可以用作构建块来构建更大的协议（包括DI位承诺和DI安全多方计算），同时仍然保持正确性和安全性保证。   文献中已知的OT DI协议都不是针对联合量子攻击的稳健且安全的。这是我们解决的与设备无关的双方不信任加密技术中的一个主要悬而未决的问题。   我们用混合（量子经典）策略证明了一类纠缠游戏的并行重复定理，以表明我们协议的安全性。混合策略有助于将DELAY纳入我们的协议中。这个平行重复定理是我们工作的主要技术贡献。



## **17. Exploring Adversarial Transferability between Kolmogorov-arnold Networks**

探索Kolmogorov-Arnold网络之间的对抗可移植性 cs.CV

After the submission of the paper, we realized that the study still  has room for expansion. In order to make the research findings more profound  and comprehensive, we have decided to withdraw the paper so that we can  conduct further research and expansion

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2503.06276v2) [paper-pdf](http://arxiv.org/pdf/2503.06276v2)

**Authors**: Songping Wang, Xinquan Yue, Yueming Lyu, Caifeng Shan

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a transformative model paradigm, significantly impacting various fields. However, their adversarial robustness remains less underexplored, especially across different KAN architectures. To explore this critical safety issue, we conduct an analysis and find that due to overfitting to the specific basis functions of KANs, they possess poor adversarial transferability among different KANs. To tackle this challenge, we propose AdvKAN, the first transfer attack method for KANs. AdvKAN integrates two key components: 1) a Breakthrough-Defense Surrogate Model (BDSM), which employs a breakthrough-defense training strategy to mitigate overfitting to the specific structures of KANs. 2) a Global-Local Interaction (GLI) technique, which promotes sufficient interaction between adversarial gradients of hierarchical levels, further smoothing out loss surfaces of KANs. Both of them work together to enhance the strength of transfer attack among different KANs. Extensive experimental results on various KANs and datasets demonstrate the effectiveness of AdvKAN, which possesses notably superior attack capabilities and deeply reveals the vulnerabilities of KANs. Code will be released upon acceptance.

摘要: Kolmogorov-Arnold Networks（KAN）已成为一种变革性模型范式，对各个领域产生了显着影响。然而，它们的对抗鲁棒性仍然没有得到充分的探索，尤其是在不同的KAN架构中。为了探索这个关键的安全问题，我们进行了分析，发现由于对KAN特定基本功能的过度匹配，它们在不同KAN之间具有较差的对抗转移性。为了应对这一挑战，我们提出了AdvKAN，这是KAN的第一种传输攻击方法。AdvKAN集成了两个关键组件：1）突破性防御代理模型（BDSM），该模型采用突破性防御训练策略来缓解对KAN特定结构的过度适应。2)一种全球-局部相互作用（GLI）技术，可促进分层水平的对抗梯度之间的充分相互作用，进一步平滑KAN的损失表面。两者共同努力，增强不同KAN之间的转会攻击力度。对各种KAN和数据集的大量实验结果证明了AdvKAN的有效性，它具有显着卓越的攻击能力，并深刻揭示了KAN的漏洞。代码将在接受后发布。



## **18. Fast Adversarial Training with Weak-to-Strong Spatial-Temporal Consistency in the Frequency Domain on Videos**

视频频域中具有弱到强时空一致性的快速对抗训练 cs.CV

After the submission of the paper, we realized that the study still  has room for expansion. In order to make the research findings more profound  and comprehensive, we have decided to withdraw the paper so that we can  conduct further research and expansion

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14921v2) [paper-pdf](http://arxiv.org/pdf/2504.14921v2)

**Authors**: Songping Wang, Hanqing Liu, Yueming Lyu, Xiantao Hu, Ziwen He, Wei Wang, Caifeng Shan, Liang Wang

**Abstract**: Adversarial Training (AT) has been shown to significantly enhance adversarial robustness via a min-max optimization approach. However, its effectiveness in video recognition tasks is hampered by two main challenges. First, fast adversarial training for video models remains largely unexplored, which severely impedes its practical applications. Specifically, most video adversarial training methods are computationally costly, with long training times and high expenses. Second, existing methods struggle with the trade-off between clean accuracy and adversarial robustness. To address these challenges, we introduce Video Fast Adversarial Training with Weak-to-Strong consistency (VFAT-WS), the first fast adversarial training method for video data. Specifically, VFAT-WS incorporates the following key designs: First, it integrates a straightforward yet effective temporal frequency augmentation (TF-AUG), and its spatial-temporal enhanced form STF-AUG, along with a single-step PGD attack to boost training efficiency and robustness. Second, it devises a weak-to-strong spatial-temporal consistency regularization, which seamlessly integrates the simpler TF-AUG and the more complex STF-AUG. Leveraging the consistency regularization, it steers the learning process from simple to complex augmentations. Both of them work together to achieve a better trade-off between clean accuracy and robustness. Extensive experiments on UCF-101 and HMDB-51 with both CNN and Transformer-based models demonstrate that VFAT-WS achieves great improvements in adversarial robustness and corruption robustness, while accelerating training by nearly 490%.

摘要: 对抗训练（AT）已被证明可以通过最小-最大优化方法显着增强对抗鲁棒性。然而，它在视频识别任务中的有效性受到两个主要挑战的阻碍。首先，视频模型的快速对抗训练在很大程度上尚未开发，这严重阻碍了其实际应用。具体来说，大多数视频对抗训练方法的计算成本很高，训练时间长，费用高。其次，现有的方法很难在清晰的准确性和对抗性稳健性之间做出权衡。为了应对这些挑战，我们引入了具有弱到强一致性的视频快速对抗训练（VFAT-WS），这是第一种针对视频数据的快速对抗训练方法。具体来说，VFAT-WS结合了以下关键设计：首先，它集成了简单而有效的时间频率增强（TF-AUG）及其时空增强形式STF-AUG，以及一步PVD攻击，以提高训练效率和鲁棒性。其次，它设计了从弱到强的时空一致性规范化，无缝集成了更简单的TF-AUG和更复杂的STF-AUG。利用一致性规范化，它将学习过程从简单扩展转向复杂扩展。两者共同努力，在干净的准确性和稳健性之间实现更好的权衡。使用CNN和基于Transformer的模型在UCF-101和HMDB-51上进行的广泛实验表明，VFAT-WS在对抗稳健性和腐败稳健性方面实现了巨大改进，同时将训练速度加快了近490%。



## **19. Information Leakage of Sentence Embeddings via Generative Embedding Inversion Attacks**

生成嵌入倒置攻击导致句子嵌入的信息泄露 cs.IR

This is a preprint of our paper accepted at SIGIR 2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16609v1) [paper-pdf](http://arxiv.org/pdf/2504.16609v1)

**Authors**: Antonios Tragoudaras, Theofanis Aslanidis, Emmanouil Georgios Lionis, Marina Orozco González, Panagiotis Eustratiadis

**Abstract**: Text data are often encoded as dense vectors, known as embeddings, which capture semantic, syntactic, contextual, and domain-specific information. These embeddings, widely adopted in various applications, inherently contain rich information that may be susceptible to leakage under certain attacks. The GEIA framework highlights vulnerabilities in sentence embeddings, demonstrating that they can reveal the original sentences they represent. In this study, we reproduce GEIA's findings across various neural sentence embedding models. Additionally, we contribute new analysis to examine whether these models leak sensitive information from their training datasets. We propose a simple yet effective method without any modification to the attacker's architecture proposed in GEIA. The key idea is to examine differences between log-likelihood for masked and original variants of data that sentence embedding models have been pre-trained on, calculated on the embedding space of the attacker. Our findings indicate that following our approach, an adversary party can recover meaningful sensitive information related to the pre-training knowledge of the popular models used for creating sentence embeddings, seriously undermining their security. Our code is available on: https://github.com/taslanidis/GEIA

摘要: 文本数据通常被编码为密集载体，称为嵌入，它捕获语义、语法、上下文和特定领域的信息。这些嵌入在各种应用程序中广泛采用，本质上包含丰富的信息，这些信息在某些攻击下可能容易泄露。GEIA框架强调了句子嵌入中的漏洞，证明它们可以揭示它们所代表的原始句子。在这项研究中，我们在各种神经句子嵌入模型中重现了GEIA的发现。此外，我们还提供新的分析来检查这些模型是否从其训练数据集中泄露敏感信息。我们提出了一种简单而有效的方法，无需对GEIA中提出的攻击者架构进行任何修改。关键想法是检查句子嵌入模型预先训练的数据的掩蔽变体和原始变体之间的日志似然性差异，并在攻击者的嵌入空间上计算。我们的研究结果表明，按照我们的方法，对手方可以恢复与用于创建句子嵌入的流行模型的预训练知识相关的有意义的敏感信息，从而严重损害了它们的安全性。我们的代码可在：https://github.com/taslanidis/GEIA



## **20. Seeking Flat Minima over Diverse Surrogates for Improved Adversarial Transferability: A Theoretical Framework and Algorithmic Instantiation**

在不同代理人上寻求平坦极小值以提高对抗可移植性：理论框架和数学实例 cs.CR

26 pages, 6 figures

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16474v1) [paper-pdf](http://arxiv.org/pdf/2504.16474v1)

**Authors**: Meixi Zheng, Kehan Wu, Yanbo Fan, Rui Huang, Baoyuan Wu

**Abstract**: The transfer-based black-box adversarial attack setting poses the challenge of crafting an adversarial example (AE) on known surrogate models that remain effective against unseen target models. Due to the practical importance of this task, numerous methods have been proposed to address this challenge. However, most previous methods are heuristically designed and intuitively justified, lacking a theoretical foundation. To bridge this gap, we derive a novel transferability bound that offers provable guarantees for adversarial transferability. Our theoretical analysis has the advantages of \textit{(i)} deepening our understanding of previous methods by building a general attack framework and \textit{(ii)} providing guidance for designing an effective attack algorithm. Our theoretical results demonstrate that optimizing AEs toward flat minima over the surrogate model set, while controlling the surrogate-target model shift measured by the adversarial model discrepancy, yields a comprehensive guarantee for AE transferability. The results further lead to a general transfer-based attack framework, within which we observe that previous methods consider only partial factors contributing to the transferability. Algorithmically, inspired by our theoretical results, we first elaborately construct the surrogate model set in which models exhibit diverse adversarial vulnerabilities with respect to AEs to narrow an instantiated adversarial model discrepancy. Then, a \textit{model-Diversity-compatible Reverse Adversarial Perturbation} (DRAP) is generated to effectively promote the flatness of AEs over diverse surrogate models to improve transferability. Extensive experiments on NIPS2017 and CIFAR-10 datasets against various target models demonstrate the effectiveness of our proposed attack.

摘要: 基于传输的黑匣子对抗攻击设置带来了在已知代理模型上制作对抗示例（AE）的挑战，该模型对未见的目标模型仍然有效。由于这项任务的实际重要性，人们提出了许多方法来应对这一挑战。然而，以前的大多数方法都是经验主义设计和直观合理的，缺乏理论基础。为了弥合这一差距，我们推导出一个新颖的可转让性界限，为对抗性可转让性提供可证明的保证。我们的理论分析的优点是：\textit{（i）}通过构建通用攻击框架加深了我们对先前方法的理解，并且\textit{（ii）}为设计有效的攻击算法提供指导。我们的理论结果表明，在代理模型集中将AE优化为平坦最小值，同时控制由对抗模型差异衡量的代理目标模型漂移，可以全面保证AE的可移植性。结果进一步导致了一个通用的基于转移的攻击框架，在该框架中，我们观察到之前的方法只考虑了影响可转移性的部分因素。从数学上来说，受我们理论结果的启发，我们首先精心构建代理模型集，其中模型表现出关于AE的各种对抗脆弱性，以缩小实例化的对抗模型差异。然后，生成\textit{model-Diversity-compatible Reversarial Perturbation}（DRAP），以有效地促进AE相对于不同代理模型的平坦性，以提高可移植性。针对各种目标模型在NIPS 2017和CIFAR-10数据集上进行的大量实验证明了我们提出的攻击的有效性。



## **21. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

用于机器学习文本分类器的自动可信度Oracle生成 cs.SE

24 pages, 5 tables, 9 figures, Camera-ready version accepted to FSE  2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2410.22663v3) [paper-pdf](http://arxiv.org/pdf/2410.22663v3)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.

摘要: 用于文本分类的机器学习（ML）已广泛应用于各个领域。这些应用程序可能会显着影响道德、经济和人类行为，从而引发人们对信任ML决策的严重担忧。研究表明，传统指标不足以建立人类对ML模型的信任。这些模型经常学习虚假相关性并基于它们进行预测。在现实世界中，他们的表现可能会显着恶化。为了避免这种情况，常见的做法是根据数据中的有效模式测试预测是否合理。与此同时，还引入了一个称为可信度Oracle问题的挑战。由于缺乏自动可信度预言，评估需要对解释方法披露的决策过程进行手动验证。然而，这耗时、容易出错且不可扩展。   我们提出了TOKI，这是第一种文本分类器的自动可信Oracle生成方法。TOKI自动检查对预测贡献最大的单词是否与预测的类别在语义上相关。具体来说，我们利用ML解释来提取影响决策的单词，并基于单词嵌入来测量它们与类的语义相关性。我们还引入了一种新颖的对抗攻击方法，该方法针对TOKI识别的可信度漏洞。为了评估它们与人类判断的一致性，我们进行了实验。我们将TOKI与仅基于模型置信度和TOKI引导的对抗攻击方法与A2 T（一种SOTA对抗攻击方法）进行比较。结果表明，依赖预测不确定性无法有效区分可信和不可信的预测，TOKI的准确性比原始基线高出142%，TOKI引导的攻击方法比A2 T更有效，干扰更少。



## **22. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

大型语言模型Sentinel：对抗性纯化的LLM代理 cs.CL

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2405.20770v4) [paper-pdf](http://arxiv.org/pdf/2405.20770v4)

**Authors**: Guang Lin, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.

摘要: 在过去的两年里，大型语言模型（LLM）的使用迅速发展。虽然这些LLM提供了相当大的便利，但它们也引发了安全问题，因为LLM很容易受到一些精心设计的文本扰动的对抗攻击。本文中，我们介绍了一种名为Large LAnguage MOdel Sentinel（LLAMOS）的新型防御技术，该技术旨在通过在将对抗性文本示例输入目标LLM之前对其进行纯化来增强LLM的对抗性鲁棒性。我们的方法包括两个主要部分：a）代理指令，可以模拟新代理进行对抗性防御，改变最小字符以保持句子的原始含义，同时防御攻击; b）防御指导，提供修改干净或对抗性示例的策略，以确保有效的防御和目标LLM的准确输出。值得注意的是，即使没有从对抗性例子中学习，防御代理也表现出强大的防御能力。此外，我们还进行了一项有趣的对抗实验，我们开发了两个代理人，一个用于防御，一个用于攻击，并让它们进行相互对抗。在对抗互动过程中，两个主体都没有完全击败对方。开源和闭源LLM上的大量实验表明，我们的方法可以有效地防御对抗攻击，从而增强对抗鲁棒性。



## **23. Property-Preserving Hashing for $\ell_1$-Distance Predicates: Applications to Countering Adversarial Input Attacks**

$\ell_1$-距离预测的属性保留哈希：对抗对抗输入攻击的应用 cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16355v1) [paper-pdf](http://arxiv.org/pdf/2504.16355v1)

**Authors**: Hassan Asghar, Chenhan Zhang, Dali Kaafar

**Abstract**: Perceptual hashing is used to detect whether an input image is similar to a reference image with a variety of security applications. Recently, they have been shown to succumb to adversarial input attacks which make small imperceptible changes to the input image yet the hashing algorithm does not detect its similarity to the original image. Property-preserving hashing (PPH) is a recent construct in cryptography, which preserves some property (predicate) of its inputs in the hash domain. Researchers have so far shown constructions of PPH for Hamming distance predicates, which, for instance, outputs 1 if two inputs are within Hamming distance $t$. A key feature of PPH is its strong correctness guarantee, i.e., the probability that the predicate will not be correctly evaluated in the hash domain is negligible. Motivated by the use case of detecting similar images under adversarial setting, we propose the first PPH construction for an $\ell_1$-distance predicate. Roughly, this predicate checks if the two one-sided $\ell_1$-distances between two images are within a threshold $t$. Since many adversarial attacks use $\ell_2$-distance (related to $\ell_1$-distance) as the objective function to perturb the input image, by appropriately choosing the threshold $t$, we can force the attacker to add considerable noise to evade detection, and hence significantly deteriorate the image quality. Our proposed scheme is highly efficient, and runs in time $O(t^2)$. For grayscale images of size $28 \times 28$, we can evaluate the predicate in $0.0784$ seconds when pixel values are perturbed by up to $1 \%$. For larger RGB images of size $224 \times 224$, by dividing the image into 1,000 blocks, we achieve times of $0.0128$ seconds per block for $1 \%$ change, and up to $0.2641$ seconds per block for $14\%$ change.

摘要: 感知散列用于检测输入图像是否与各种安全应用中的参考图像相似。最近，它们已被证明屈服于对抗性输入攻击，这些攻击对输入图像进行微小的不可感知的更改，但哈希算法无法检测其与原始图像的相似性。保属性散列（Property Preserving Hashing，PPH）是密码学中的一种新构造，它在散列域中保留了其输入的某些属性（谓词）。到目前为止，研究人员已经展示了PPH的汉明距离谓词的构造，例如，如果两个输入在汉明距离$t$内，则输出1。PPH的一个关键特征是其强大的正确性保证，即在哈希域中不正确评估该断言的可能性可以忽略不计。出于在对抗环境下检测相似图像的用例的动机，我们提出了$\ell_1 $-距离断言的第一个PPH结构。大致上，此断言检查两个图像之间的两个单边$\ell_1$-距离是否在阈值$t$内。由于许多对抗性攻击使用$\ell_2$-距离（与$\ell_1$-距离相关）作为干扰输入图像的目标函数，因此通过适当选择阈值$t$，我们可以迫使攻击者添加相当大的噪音来逃避检测，从而显着降低图像质量。我们提出的方案非常高效，并且在时间内运行$O（t#2）$。对于大小为$28 \x 28$的灰度图像，当像素值受到高达$1 \%$的干扰时，我们可以在$0.0784$秒内评估该预测。对于大小为$224 \x 224\$的较大Ruby图像，通过将图像分为1，000个块，我们可以实现每个块0.0128 $秒的时间（$1 \%$改变），每个块最多0.2641 $秒的时间（$14\%$改变）。



## **24. Blockchain Meets Adaptive Honeypots: A Trust-Aware Approach to Next-Gen IoT Security**

区块链遇上自适应蜜罐：下一代物联网安全的信任意识方法 cs.CR

This paper has been submitted to the IEEE Transactions on Network  Science and Engineering (TNSE) for possible publication

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.16226v1) [paper-pdf](http://arxiv.org/pdf/2504.16226v1)

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak

**Abstract**: Edge computing-based Next-Generation Wireless Networks (NGWN)-IoT offer enhanced bandwidth capacity for large-scale service provisioning but remain vulnerable to evolving cyber threats. Existing intrusion detection and prevention methods provide limited security as adversaries continually adapt their attack strategies. We propose a dynamic attack detection and prevention approach to address this challenge. First, blockchain-based authentication uses the Deoxys Authentication Algorithm (DAA) to verify IoT device legitimacy before data transmission. Next, a bi-stage intrusion detection system is introduced: the first stage uses signature-based detection via an Improved Random Forest (IRF) algorithm. In contrast, the second stage applies feature-based anomaly detection using a Diffusion Convolution Recurrent Neural Network (DCRNN). To ensure Quality of Service (QoS) and maintain Service Level Agreements (SLA), trust-aware service migration is performed using Heap-Based Optimization (HBO). Additionally, on-demand virtual High-Interaction honeypots deceive attackers and extract attack patterns, which are securely stored using the Bimodal Lattice Signature Scheme (BLISS) to enhance signature-based Intrusion Detection Systems (IDS). The proposed framework is implemented in the NS3 simulation environment and evaluated against existing methods across multiple performance metrics, including accuracy, attack detection rate, false negative rate, precision, recall, ROC curve, memory usage, CPU usage, and execution time. Experimental results demonstrate that the framework significantly outperforms existing approaches, reinforcing the security of NGWN-enabled IoT ecosystems

摘要: 基于边缘计算的下一代无线网络（NGWN）-物联网为大规模服务提供提供增强的带宽容量，但仍然容易受到不断变化的网络威胁的影响。由于对手不断调整其攻击策略，现有的入侵检测和预防方法提供的安全性有限。我们提出了一种动态攻击检测和预防方法来应对这一挑战。首先，基于区块链的认证使用Deoxys认证算法（DAA）来在数据传输之前验证物联网设备的合法性。接下来，介绍了一个两阶段的入侵检测系统：第一阶段使用基于签名的检测通过改进的随机森林（IRF）算法。相比之下，第二阶段应用基于特征的异常检测使用扩散卷积递归神经网络（DCRNN）。为了确保服务质量（QoS）和维护服务水平协议（SLA），使用基于堆的优化（HBO）来执行信任感知服务迁移。此外，按需虚拟高交互蜜罐欺骗攻击者并提取攻击模式，这些模式使用双峰格签名方案（BLISS）安全存储，以增强基于签名的入侵检测系统（IDS）。所提出的框架在NS 3模拟环境中实施，并针对多个性能指标（包括准确性、攻击检测率、误报率、精确性、召回率、ROC曲线、内存使用率、中央处理器使用率和执行时间）的现有方法进行评估。实验结果表明，该框架的性能显着优于现有方法，增强了NGWN支持的物联网生态系统的安全性



## **25. Adversarial Observations in Weather Forecasting**

天气预报中的对抗性观测 cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15942v1) [paper-pdf](http://arxiv.org/pdf/2504.15942v1)

**Authors**: Erik Imgrund, Thorsten Eisenhofer, Konrad Rieck

**Abstract**: AI-based systems, such as Google's GenCast, have recently redefined the state of the art in weather forecasting, offering more accurate and timely predictions of both everyday weather and extreme events. While these systems are on the verge of replacing traditional meteorological methods, they also introduce new vulnerabilities into the forecasting process. In this paper, we investigate this threat and present a novel attack on autoregressive diffusion models, such as those used in GenCast, capable of manipulating weather forecasts and fabricating extreme events, including hurricanes, heat waves, and intense rainfall. The attack introduces subtle perturbations into weather observations that are statistically indistinguishable from natural noise and change less than 0.1% of the measurements - comparable to tampering with data from a single meteorological satellite. As modern forecasting integrates data from nearly a hundred satellites and many other sources operated by different countries, our findings highlight a critical security risk with the potential to cause large-scale disruptions and undermine public trust in weather prediction.

摘要: 谷歌的GenCast等基于人工智能的系统最近重新定义了天气预报的最新水平，为日常天气和极端事件提供了更准确、更及时的预测。虽然这些系统即将取代传统的气象方法，但它们也给预报过程带来了新的漏洞。在本文中，我们调查了这一威胁，并对自回归扩散模型（例如GenCast中使用的模型）提出了一种新颖的攻击，这些模型能够操纵天气预报并编造极端事件，包括飓风、热浪和强降雨。该攻击在天气观测中引入了微妙的扰动，这些扰动在统计上与自然噪音难以区分，并且测量结果的变化不到0.1%--与篡改单个气象卫星的数据相当。由于现代预报整合了来自不同国家运营的近一百颗卫星和许多其他来源的数据，我们的研究结果凸显了一个关键的安全风险，有可能造成大规模干扰并破坏公众对天气预测的信任。



## **26. Human-Imperceptible Physical Adversarial Attack for NIR Face Recognition Models**

近红外人脸识别模型的人类不可感知的物理对抗攻击 cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15823v1) [paper-pdf](http://arxiv.org/pdf/2504.15823v1)

**Authors**: Songyan Xie, Jinghang Wen, Encheng Su, Qiucheng Yu

**Abstract**: Near-infrared (NIR) face recognition systems, which can operate effectively in low-light conditions or in the presence of makeup, exhibit vulnerabilities when subjected to physical adversarial attacks. To further demonstrate the potential risks in real-world applications, we design a novel, stealthy, and practical adversarial patch to attack NIR face recognition systems in a black-box setting. We achieved this by utilizing human-imperceptible infrared-absorbing ink to generate multiple patches with digitally optimized shapes and positions for infrared images. To address the optimization mismatch between digital and real-world NIR imaging, we develop a light reflection model for human skin to minimize pixel-level discrepancies by simulating NIR light reflection.   Compared to state-of-the-art (SOTA) physical attacks on NIR face recognition systems, the experimental results show that our method improves the attack success rate in both digital and physical domains, particularly maintaining effectiveness across various face postures. Notably, the proposed approach outperforms SOTA methods, achieving an average attack success rate of 82.46% in the physical domain across different models, compared to 64.18% for existing methods. The artifact is available at https://anonymous.4open.science/r/Human-imperceptible-adversarial-patch-0703/.

摘要: 近红外（NIR）人脸识别系统可以在弱光条件下或化妆时有效工作，但在受到物理对抗攻击时会表现出脆弱性。为了进一步证明现实世界应用程序中的潜在风险，我们设计了一种新颖、隐蔽且实用的对抗补丁来攻击黑匣子环境中的近红外人脸识别系统。我们通过利用人类难以感知的红外吸收墨水来生成具有数字优化的红外图像形状和位置的多个补丁来实现这一目标。为了解决数字和现实世界的近红外成像之间的优化不匹配问题，我们开发了一种用于人类皮肤的光反射模型，通过模拟近红外光反射来最大限度地减少像素级差异。   与对近红外人脸识别系统的最新技术（SOTA）物理攻击相比，实验结果表明，我们的方法提高了数字和物理领域的攻击成功率，特别是在各种面部姿势下保持有效性。值得注意的是，提出的方法优于SOTA方法，在不同模型的物理域中实现了82.46%的平均攻击成功率，而现有方法的平均攻击成功率为64.18%。该产品可在https://anonymous.4open.science/r/Human-imperceptible-adversarial-patch-0703/上找到。



## **27. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

下一代物联网的图形神经网络：最近的进展和开放的挑战 cs.IT

28 pages, 15 figures, and 6 tables. Submitted for publication

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2412.20634v2) [paper-pdf](http://arxiv.org/pdf/2412.20634v2)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a critical tool for optimizing and managing the complexities of the Internet of Things (IoT) in next-generation networks. This survey presents a comprehensive exploration of how GNNs may be harnessed in 6G IoT environments, focusing on key challenges and opportunities through a series of open questions. We commence with an exploration of GNN paradigms and the roles of node, edge, and graph-level tasks in solving wireless networking problems and highlight GNNs' ability to overcome the limitations of traditional optimization methods. This guidance enhances problem-solving efficiency across various next-generation (NG) IoT scenarios. Next, we provide a detailed discussion of the application of GNN in advanced NG enabling technologies, including massive MIMO, reconfigurable intelligent surfaces, satellites, THz, mobile edge computing (MEC), and ultra-reliable low latency communication (URLLC). We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms to secure GNN-based NG-IoT networks. Next, we examine how GNNs can be integrated with future technologies like integrated sensing and communication (ISAC), satellite-air-ground-sea integrated networks (SAGSIN), and quantum computing. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security within NG-IoT systems, paving the way for future advances. Finally, we propose a set of design guidelines to facilitate the development of efficient, scalable, and secure GNN models tailored for NG IoT applications.

摘要: 图形神经网络（GNN）已成为优化和管理下一代网络中物联网（IoT）复杂性的重要工具。这项调查全面探索了如何在6 G物联网环境中利用GNN，并通过一系列开放性问题重点关注关键挑战和机遇。我们首先探索GNN范式以及节点、边缘和图形级任务在解决无线网络问题中的作用，并强调GNN克服传统优化方法局限性的能力。该指南提高了各种下一代（NG）物联网场景中的问题解决效率。接下来，我们详细讨论GNN在先进NG使能技术中的应用，包括大规模MMO、可重配置智能表面、卫星、太赫兹、移动边缘计算（MEC）和超可靠低延迟通信（URLLC）。然后，我们深入研究了对抗性攻击带来的挑战，深入了解保护基于GNN的NG-物联网网络的防御机制。接下来，我们研究GNN如何与集成传感和通信（ISAC）、卫星-空-地-海综合网络（SAGSIN）和量子计算等未来技术集成。我们的研究结果强调了GNN在提高NG-物联网系统内的效率、可扩展性和安全性方面的变革潜力，为未来的发展铺平了道路。最后，我们提出了一套设计准则，以促进开发为NG物联网应用量身定制的高效、可扩展且安全的GNN模型。



## **28. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多模式大型语言模型（MLLM）在各种多模式任务中展示了令人印象深刻的性能。另一方面，额外图像形态的集成可能会允许恶意用户在图像中注入有害内容以进行越狱。与基于文本的LLM不同，对手需要选择离散令牌来使用特定算法隐藏其恶意意图，图像信号的连续性为对手提供了注入有害意图的直接机会。在这项工作中，我们提出了$\textBF{BaThe}$（$\textBF{BA}$ckdoor $\textBF{T}$rigger S$\textBF{h}$i$\textBF{e}$ld），这是一种简单而有效的越狱防御机制。我们的工作受到最近对生成性语言模型中越狱后门攻击和虚拟提示后门攻击的研究的启发。越狱后门攻击使用有害指令与手工制作的字符串相结合作为触发器，使后门模型生成禁止的响应。我们假设有害指令可以充当触发器，如果我们将拒绝响应设置为触发响应，那么后门模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一目标，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为“wedge”。我们的全面实验表明，BaThe有效地缓解了各种类型的越狱攻击，并且能够抵御不可见的攻击，对MLLM的性能影响最小。



## **29. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

Red Team Diffuser：通过强化学习暴露视觉语言模型中的有毒连续漏洞 cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.

摘要: 大型视觉语言模型（VLM）的不断增加的部署暴露了其对齐机制中的关键安全漏洞。虽然现有的越狱研究主要关注VLM对有害指令的敏感性，但我们揭示了一个基本但被忽视的弱点：有毒文本延续，当提示有害文本前置与语义对抗图像配对时，VLM会产生剧毒的完成。为了系统性地研究这种威胁，我们提出了Red Team Distuser（RTI），这是第一个红色团队扩散模型，通过强化学习协调对抗图像生成和有毒延续。我们的关键创新包括动态跨模式攻击和隐身优化。对于LLM安全基准中的有毒文本前置，我们进行贪婪搜索以识别最大限度地引发有毒完成的最佳图像提示。发现的图像提示然后驱动基于RL的扩散模型微调，产生语义对齐的对抗图像，从而提高毒性率。潜行感知优化引入了联合对抗奖励，平衡毒性最大化（通过Dealfy分类器）和潜行性（通过BERTScore），规避了传统的基于噪音的对抗模式。实验结果证明了RTI的有效性，在原始攻击集中，LLaVA输出的毒性率比纯文本基线增加了10.69%，在未见集上增加了8.91%，证明了概括能力。此外，RTI具有较强的跨模型转移性，使Gemini的毒性率提高了5.1%，对LLaMA的毒性率提高了26.83%。我们的研究结果暴露了当前VLM对齐中的两个关键缺陷：（1）未能防止有害前置的有毒延续，以及（2）忽视了跨模式攻击载体。这些结果需要在安全性评估中向多模式红色团队转变。



## **30. TrojanDam: Detection-Free Backdoor Defense in Federated Learning through Proactive Model Robustification utilizing OOD Data**

TrojanDam：通过利用OOD数据的主动模型Robusification在联邦学习中实现无检测后门防御 cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15674v1) [paper-pdf](http://arxiv.org/pdf/2504.15674v1)

**Authors**: Yanbo Dai, Songze Li, Zihan Gan, Xueluan Gong

**Abstract**: Federated learning (FL) systems allow decentralized data-owning clients to jointly train a global model through uploading their locally trained updates to a centralized server. The property of decentralization enables adversaries to craft carefully designed backdoor updates to make the global model misclassify only when encountering adversary-chosen triggers. Existing defense mechanisms mainly rely on post-training detection after receiving updates. These methods either fail to identify updates which are deliberately fabricated statistically close to benign ones, or show inconsistent performance in different FL training stages. The effect of unfiltered backdoor updates will accumulate in the global model, and eventually become functional. Given the difficulty of ruling out every backdoor update, we propose a backdoor defense paradigm, which focuses on proactive robustification on the global model against potential backdoor attacks. We first reveal that the successful launching of backdoor attacks in FL stems from the lack of conflict between malicious and benign updates on redundant neurons of ML models. We proceed to prove the feasibility of activating redundant neurons utilizing out-of-distribution (OOD) samples in centralized settings, and migrating to FL settings to propose a novel backdoor defense mechanism, TrojanDam. The proposed mechanism has the FL server continuously inject fresh OOD mappings into the global model to activate redundant neurons, canceling the effect of backdoor updates during aggregation. We conduct systematic and extensive experiments to illustrate the superior performance of TrojanDam, over several SOTA backdoor defense methods across a wide range of FL settings.

摘要: 联合学习（FL）系统允许分散的数据拥有客户通过将本地训练的更新上传到集中式服务器来联合训练全球模型。去中心化的属性使对手能够精心设计的后门更新，以使全球模型仅在遇到对手选择的触发器时才进行错误分类。现有的防御机制主要依赖于收到更新后的训练后检测。这些方法要么无法识别在统计上故意编造的接近良性更新的更新，要么在不同的FL训练阶段表现不一致。未经过滤的后门更新的影响将在全局模型中累积，并最终成为功能。考虑到排除每一个后门更新的困难，我们提出了一个后门防御范式，它侧重于对潜在的后门攻击的全局模型的主动鲁棒性。我们首先揭示了在FL中成功发起后门攻击源于ML模型的冗余神经元上的恶意和良性更新之间缺乏冲突。我们继续证明在集中式设置中利用分布外（OOD）样本激活冗余神经元的可行性，并迁移到FL设置，提出一种新的后门防御机制TrojanDam。所提出的机制让FL服务器不断地将新的OOD映射注入到全局模型中，以激活冗余神经元，从而消除聚合期间后门更新的影响。我们进行了系统且广泛的实验，以说明TrojanDam在各种FL设置中优于多种SOTA后门防御方法的卓越性能。



## **31. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

通过跨模式提示注射操纵多模式代理 cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.14348v2) [paper-pdf](http://arxiv.org/pdf/2504.14348v2)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.

摘要: 多模式大型语言模型的出现通过将语言和视觉模式与外部数据源集成来重新定义了代理范式，使代理能够更好地解释人类指令并执行日益复杂的任务。然而，在这项工作中，我们发现了多模式代理中一个以前被忽视的关键安全漏洞：跨模式提示注入攻击。为了利用这个漏洞，我们提出了CrossInib，这是一种新型攻击框架，其中攻击者在多种模式中嵌入对抗性扰动，以与目标恶意内容保持一致，允许外部指令劫持代理的决策过程并执行未经授权的任务。我们的方法由两个关键部分组成。首先，我们引入了视觉潜在对齐，基于文本到图像生成模型，优化视觉嵌入空间中恶意指令的对抗特征，确保对抗图像巧妙地编码恶意任务执行的线索。随后，我们提出了文本指导增强，其中利用大型语言模型通过对抗性Meta提示来推断黑匣子防御系统提示，并生成恶意文本命令，该命令引导代理的输出更好地遵守攻击者的请求。大量实验表明，我们的方法优于现有的注入攻击，在不同任务中的攻击成功率至少增加了+26.4%。此外，我们还验证了攻击在现实世界的多模式自治代理中的有效性，强调了其对安全关键应用程序的潜在影响。



## **32. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

Gungnir：利用图像中的风格特征对扩散模型进行后门攻击 cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.20650v2) [paper-pdf](http://arxiv.org/pdf/2502.20650v2)

**Authors**: Yu Pan, Bingrong Dai, Jiahao Chen, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.

摘要: 近年来，扩散模型（DM）在图像生成领域取得了重大进展。然而，根据当前的研究，DM很容易受到后门攻击，后门攻击允许攻击者通过输入包含隐蔽触发器（例如特定的视觉补丁或短语）的数据来控制模型的输出。现有的防御策略完全可以通过后门检测和触发器倒置来阻止此类攻击，因为以前的攻击方法受到有限的输入空间和低维触发器的限制。例如，视觉触发器很容易被防御者观察到，基于文本或基于注意力的触发器更容易受到神经网络检测的影响。为了探索DM中后门攻击的更多可能性，我们提出了Gungnir，这是一种新颖的方法，使攻击者能够通过输入图像中的风格触发器激活DM中的后门。我们的方法首次提出使用风格特征作为触发器，并通过引入重建对抗噪音（RAN）和短期时间间隔保留（STTR）在图像到图像任务中成功实施后门攻击。我们的技术生成的嵌入式图像在感知上与干净图像无法区分，从而绕过了手动检查和自动检测神经网络。实验表明，贡尼尔可以轻松绕过现有的防御方法。在现有的DM防御框架中，我们的方法实现了0后门检测率（BDR）。我们的代码可在https://github.com/paoche11/Gungnir上获取。



## **33. Evaluating the Robustness of Multimodal Agents Against Active Environmental Injection Attacks**

多模态Agent对主动环境注入攻击的鲁棒性评估 cs.CL

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.13053v2) [paper-pdf](http://arxiv.org/pdf/2502.13053v2)

**Authors**: Yurun Chen, Xavier Hu, Keting Yin, Juncheng Li, Shengyu Zhang

**Abstract**: As researchers continue to optimize AI agents for more effective task execution within operating systems, they often overlook a critical security concern: the ability of these agents to detect "impostors" within their environment. Through an analysis of the agents' operational context, we identify a significant threat-attackers can disguise malicious attacks as environmental elements, injecting active disturbances into the agents' execution processes to manipulate their decision-making. We define this novel threat as the Active Environment Injection Attack (AEIA). Focusing on the interaction mechanisms of the Android OS, we conduct a risk assessment of AEIA and identify two critical security vulnerabilities: (1) Adversarial content injection in multimodal interaction interfaces, where attackers embed adversarial instructions within environmental elements to mislead agent decision-making; and (2) Reasoning gap vulnerabilities in the agent's task execution process, which increase susceptibility to AEIA attacks during reasoning. To evaluate the impact of these vulnerabilities, we propose AEIA-MN, an attack scheme that exploits interaction vulnerabilities in mobile operating systems to assess the robustness of MLLM-based agents. Experimental results show that even advanced MLLMs are highly vulnerable to this attack, achieving a maximum attack success rate of 93% on the AndroidWorld benchmark by combining two vulnerabilities.

摘要: 随着研究人员不断优化人工智能代理，以便在操作系统中更有效地执行任务，他们往往忽视了一个关键的安全问题：这些代理在其环境中检测“冒名顶替者”的能力。通过分析代理的操作环境，我们确定了一个重大的威胁-攻击者可以伪装成环境元素的恶意攻击，注入主动干扰代理的执行过程，操纵他们的决策。我们将这种新的威胁定义为主动环境注入攻击（AEIA）。针对Android操作系统的交互机制，我们对AEIA进行了风险评估，并发现了两个关键的安全漏洞：（1）多模态交互界面中的对抗性内容注入，攻击者在环境元素中嵌入对抗性指令，以误导Agent决策;和（2）代理任务执行过程中的推理缺口漏洞，这增加了推理期间对AEIA攻击的易感性。为了评估这些漏洞的影响，我们提出了AEIA-NN，这是一种攻击方案，利用移动操作系统中的交互漏洞来评估基于MLLM的代理的稳健性。实验结果表明，即使是高级MLLM也极易受到这种攻击，通过组合两个漏洞，在AndroidWorld基准上实现了93%的最高攻击成功率。



## **34. Unifying Image Counterfactuals and Feature Attributions with Latent-Space Adversarial Attacks**

统一图像反事实和特征属性与潜在空间对抗攻击 cs.LG

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15479v1) [paper-pdf](http://arxiv.org/pdf/2504.15479v1)

**Authors**: Jeremy Goldwasser, Giles Hooker

**Abstract**: Counterfactuals are a popular framework for interpreting machine learning predictions. These what if explanations are notoriously challenging to create for computer vision models: standard gradient-based methods are prone to produce adversarial examples, in which imperceptible modifications to image pixels provoke large changes in predictions. We introduce a new, easy-to-implement framework for counterfactual images that can flexibly adapt to contemporary advances in generative modeling. Our method, Counterfactual Attacks, resembles an adversarial attack on the representation of the image along a low-dimensional manifold. In addition, given an auxiliary dataset of image descriptors, we show how to accompany counterfactuals with feature attribution that quantify the changes between the original and counterfactual images. These importance scores can be aggregated into global counterfactual explanations that highlight the overall features driving model predictions. While this unification is possible for any counterfactual method, it has particular computational efficiency for ours. We demonstrate the efficacy of our approach with the MNIST and CelebA datasets.

摘要: 反事实是解释机器学习预测的流行框架。众所周知，为计算机视觉模型创建这些假设解释具有挑战性：标准的基于梯度的方法容易产生对抗性的例子，其中对图像像素的难以察觉的修改会引发预测的巨大变化。我们为反事实图像引入了一种新的、易于实施的框架，该框架可以灵活地适应生成建模的当代进步。我们的方法“反事实攻击”类似于对沿着低维多管的图像表示的对抗攻击。此外，给定图像描述符的辅助数据集，我们展示了如何伴随反事实与特征属性，量化原始图像和反事实图像之间的变化。这些重要性分数可以聚合成全局反事实解释，突出显示驱动模型预测的整体特征。虽然这种统一对任何反事实方法都是可能的，但它对我们的方法具有特定的计算效率。我们用MNIST和CelebA数据集证明了我们方法的有效性。



## **35. An Undetectable Watermark for Generative Image Models**

生成图像模型的不可检测水印 cs.CR

ICLR 2025

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.07369v4) [paper-pdf](http://arxiv.org/pdf/2410.07369v4)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.

摘要: 我们提出了第一个用于生成图像模型的不可检测水印方案。不可检测性确保没有有效的对手能够区分带水印和未带水印的图像，即使在进行了多次自适应查询之后。特别是，在任何有效可计算的指标下，不可检测的水印不会降低图像质量。我们的方案通过使用伪随机错误纠正码（Christ和Gunn，2024）选择扩散模型的初始潜伏，这是一种保证不可检测性和鲁棒性的策略。我们通过实验证明，使用稳定扩散2.1，我们的水印具有质量保护性和鲁棒性。我们的实验证明，与我们测试的每个先前方案相比，我们的水印不会降低图像质量。我们的实验还证明了鲁棒性：现有的水印去除攻击无法在不显着降低图像质量的情况下从图像中去除水印。最后，我们发现我们可以对水印中的512位进行鲁棒性编码，当图像不受到水印删除攻击时，最多可编码2500位。我们的代码可在https://github.com/XuandongZhao/PRC-Watermark上获取。



## **36. A Framework for Evaluating Emerging Cyberattack Capabilities of AI**

评估人工智能新兴网络攻击能力的框架 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.11917v3) [paper-pdf](http://arxiv.org/pdf/2503.11917v3)

**Authors**: Mikel Rodriguez, Raluca Ada Popa, Four Flynn, Lihao Liang, Allan Dafoe, Anna Wang

**Abstract**: As frontier AI models become more capable, evaluating their potential to enable cyberattacks is crucial for ensuring the safe development of Artificial General Intelligence (AGI). Current cyber evaluation efforts are often ad-hoc, lacking systematic analysis of attack phases and guidance on targeted defenses. This work introduces a novel evaluation framework that addresses these limitations by: (1) examining the end-to-end attack chain, (2) identifying gaps in AI threat evaluation, and (3) helping defenders prioritize targeted mitigations and conduct AI-enabled adversary emulation for red teaming. Our approach adapts existing cyberattack chain frameworks for AI systems. We analyzed over 12,000 real-world instances of AI involvement in cyber incidents, catalogued by Google's Threat Intelligence Group, to curate seven representative attack chain archetypes. Through a bottleneck analysis on these archetypes, we pinpointed phases most susceptible to AI-driven disruption. We then identified and utilized externally developed cybersecurity model evaluations focused on these critical phases. We report on AI's potential to amplify offensive capabilities across specific attack stages, and offer recommendations for prioritizing defenses. We believe this represents the most comprehensive AI cyber risk evaluation framework published to date.

摘要: 随着前沿人工智能模型变得越来越强大，评估其实现网络攻击的潜力对于确保通用人工智能（AGI）的安全发展至关重要。目前的网络评估工作往往是临时性的，缺乏对攻击阶段的系统分析和对有针对性防御的指导。这项工作引入了一个新的评估框架，通过以下方式解决这些限制：（1）检查端到端攻击链，（2）识别AI威胁评估中的差距，以及（3）帮助防御者优先考虑有针对性的缓解措施，并进行支持AI的对手模拟。我们的方法将现有的网络攻击链框架应用于人工智能系统。我们分析了谷歌威胁情报小组编目的12，000多个人工智能参与网络事件的现实案例，以策划七个有代表性的攻击链原型。通过对这些原型的瓶颈分析，我们找到了最容易受到人工智能驱动破坏的阶段。然后，我们确定并利用外部开发的专注于这些关键阶段的网络安全模型评估。我们报告了人工智能在特定攻击阶段增强进攻能力的潜力，并提供了优先考虑防御的建议。我们相信这代表了迄今为止发布的最全面的人工智能网络风险评估框架。



## **37. MST3 Encryption improvement with three-parameter group of Hermitian function field**

MST 3使用Hermitian函数域的三参数组进行加密改进 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15391v1) [paper-pdf](http://arxiv.org/pdf/2504.15391v1)

**Authors**: Gennady Khalimov, Yevgen Kotukh

**Abstract**: This scholarly work presents an advanced cryptographic framework utilizing automorphism groups as the foundational structure for encryption scheme implementation. The proposed methodology employs a three-parameter group construction, distinguished by its application of logarithmic signatures positioned outside the group's center, a significant departure from conventional approaches. A key innovation in this implementation is utilizing the Hermitian function field as the underlying mathematical framework. This particular function field provides enhanced structural properties that strengthen the cryptographic protocol when integrated with the three-parameter group architecture. The encryption mechanism features phased key de-encapsulation from ciphertext, representing a substantial advantage over alternative implementations. This sequential extraction process introduces additional computational complexity for potential adversaries while maintaining efficient legitimate decryption. A notable characteristic of this cryptosystem is the direct correlation between the underlying group's mathematical strength and both the attack complexity and message size parameters. This relationship enables precise security-efficiency calibration based on specific implementation requirements and threat models. The application of automorphism groups with logarithmic signatures positioned outside the center represents a significant advancement in non-traditional cryptographic designs, particularly relevant in the context of post-quantum cryptographic resilience.

摘要: 这项学术工作提出了一个先进的加密框架，利用自同构群作为加密方案实现的基础结构。所提出的方法采用了三参数组的建设，其应用对数签名定位组的中心之外，显着偏离传统的方法。在这种实现中的一个关键创新是利用厄米函数场作为底层的数学框架。这个特定的函数字段提供增强的结构属性，当与三参数组架构集成时，该结构属性增强了加密协议。该加密机制的特点是分阶段从密文中解封装密钥，与替代实现相比具有巨大优势。这种顺序提取过程为潜在对手带来了额外的计算复杂性，同时保持高效的合法解密。该加密系统的一个显着特征是基础组的数学强度与攻击复杂性和消息大小参数之间直接相关。这种关系可以根据特定的实施要求和威胁模型进行精确的安全效率校准。对数签名位于中心之外的自同构群的应用代表了非传统密码设计的重大进步，特别是在后量子密码弹性的背景下。



## **38. MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning**

Mr. Guard：使用课程学习的多语言推理保护 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15241v1) [paper-pdf](http://arxiv.org/pdf/2504.15241v1)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.

摘要: 大型语言模型（LLM）容易受到诸如越狱之类的对抗性攻击，这可能会引发有害或不安全的行为。在多语言环境中，这种漏洞会加剧，因为多语言安全一致的数据通常有限。因此，开发一个能够检测和过滤不同语言的不安全内容的护栏对于在现实世界的应用程序中部署LLM至关重要。在这项工作中，我们提出了一种方法来建立一个多语言护栏推理。我们的方法包括：（1）综合多语言数据生成，融合了文化和语言上的细微差别，（2）监督微调，以及（3）课程引导的群组相对政策优化（GRPO）框架，进一步提高性能。实验结果表明，我们的多语言护栏在域内和域外语言中始终优于最近的基线。我们护栏的多语言推理能力使其能够生成多语言解释，这对于理解多语言内容审核中的特定语言风险和歧义特别有用。



## **39. Progressive Pruning: Analyzing the Impact of Intersection Attacks**

渐进修剪：分析交叉点攻击的影响 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.08700v2) [paper-pdf](http://arxiv.org/pdf/2410.08700v2)

**Authors**: Christoph Döpmann, Maximilian Weisenseel, Florian Tschorsch

**Abstract**: Stream-based communication dominates today's Internet, posing unique challenges for anonymous communication networks (ACNs). Traditionally designed for independent messages, ACNs struggle to account for the inherent vulnerabilities of streams, such as susceptibility to intersection attacks. In this work, we address this gap and introduce progressive pruning, a novel methodology for quantifying the susceptibility to intersection attacks. Progressive pruning quantifies and monitors anonymity sets over time, providing an assessment of an adversary's success in correlating senders and receivers. We leverage this methodology to analyze synthetic scenarios and large-scale simulations of the Tor network using our newly developed TorFS simulator. Our findings reveal that anonymity is significantly influenced by stream length, user population, and stream distribution across the network. These insights highlight critical design challenges for future ACNs seeking to safeguard stream-based communication against traffic analysis attacks.

摘要: 基于流的通信在当今的互联网中占据主导地位，给匿名通信网络（ACN）带来了独特的挑战。ACN传统上是为独立消息设计的，很难解决流的固有漏洞，例如容易受到交叉攻击。在这项工作中，我们解决了这一差距并引入了渐进式修剪，这是一种用于量化交叉攻击易感性的新颖方法。渐进式修剪随着时间的推移量化和监控匿名集，提供对对手关联收件箱和接收器的成功评估。我们利用这种方法来使用我们新开发的TorFS模拟器分析Tor网络的合成场景和大规模模拟。我们的研究结果表明，匿名性受到流长度、用户群体和整个网络的流分布的显着影响。这些见解凸显了未来ACN的关键设计挑战，这些ACN旨在保护基于流的通信免受流量分析攻击。



## **40. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

HiddenDetect：通过监视隐藏状态检测针对大型视觉语言模型的越狱攻击 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.14744v3) [paper-pdf](http://arxiv.org/pdf/2502.14744v3)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.

摘要: 与纯语言模型相比，其他模式的集成增加了大型视觉语言模型（LVLM）对安全风险（如越狱攻击）的敏感性。虽然现有的研究主要集中在事后对齐技术，LVLM内的潜在安全机制仍然在很大程度上未被探索。在这项工作中，我们调查是否LVLM内在编码安全相关的信号在其内部激活过程中的推理。我们的研究结果表明，LVLM在处理不安全的提示时表现出不同的激活模式，可以利用它来检测和减轻对抗性输入，而不需要进行广泛的微调。基于这一见解，我们引入了HiddenDetect，这是一个新颖的免调框架，可以利用内部模型激活来增强安全性。实验结果表明，{HiddenDetect}在检测针对LVLM的越狱攻击方面超越了最先进的方法。通过利用固有的安全感知模式，我们的方法提供了一种高效且可扩展的解决方案，用于加强LVLM针对多模式威胁的鲁棒性。我们的代码将在https://github.com/leigest519/HiddenDetect上公开发布。



## **41. Scalable Discrete Event Simulation Tool for Large-Scale Cyber-Physical Energy Systems: Advancing System Efficiency and Scalability**

用于大规模网络物理能源系统的可扩展离散事件模拟工具：提高系统效率和可扩展性 eess.SY

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15198v1) [paper-pdf](http://arxiv.org/pdf/2504.15198v1)

**Authors**: Khandaker Akramul Haque, Shining Sun, Xiang Huo, Ana E. Goulart, Katherine R. Davis

**Abstract**: Modern power systems face growing risks from cyber-physical attacks, necessitating enhanced resilience due to their societal function as critical infrastructures. The challenge is that defense of large-scale systems-of-systems requires scalability in their threat and risk assessment environment for cyber physical analysis including cyber-informed transmission planning, decision-making, and intrusion response. Hence, we present a scalable discrete event simulation tool for analysis of energy systems, called DESTinE. The tool is tailored for largescale cyber-physical systems, with a focus on power systems. It supports faster-than-real-time traffic generation and models packet flow and congestion under both normal and adversarial conditions. Using three well-established power system synthetic cases with 500, 2000, and 10,000 buses, we overlay a constructed cyber network employing star and radial topologies. Experiments are conducted to identify critical nodes within a communication network in response to a disturbance. The findings are incorporated into a constrained optimization problem to assess the impact of the disturbance on a specific node and its cascading effects on the overall network. Based on the solution of the optimization problem, a new hybrid network topology is also derived, combining the strengths of star and radial structures to improve network resilience. Furthermore, DESTinE is integrated with a virtual server and a hardware-in-the-loop (HIL) system using Raspberry Pi 5.

摘要: 现代电力系统面临着越来越大的网络物理攻击风险，由于其作为关键基础设施的社会功能，因此需要增强弹性。挑战在于，大规模系统的防御需要其威胁和风险评估环境的可扩展性，以进行网络物理分析，包括基于网络的传输规划、决策和入侵响应。因此，我们提出了一种用于分析能源系统的可扩展离散事件模拟工具，称为DESTinE。该工具专为大型网络物理系统量身定制，重点关注电力系统。它支持比实时更快的流量生成，并对正常和对抗条件下的分组流和拥堵进行建模。我们使用三个成熟的电力系统合成案例（包含500、2000和10，000条公交车），覆盖了采用星型和放射状布局的构建网络网络。进行实验来识别通信网络内的关键节点以响应干扰。将研究结果纳入约束优化问题中，以评估干扰对特定节点的影响及其对整个网络的级联影响。在解决优化问题的基础上，还推导出了一种新的混合网络布局，结合星型和辐射型结构的优点，提高网络弹性。此外，DESTinE还使用Raspberry Pi 5与虚拟服务器和硬件在环（HIL）系统集成。



## **42. RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search**

RainbowPlus：通过进化质量多样性搜索增强对抗提示生成 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15047v1) [paper-pdf](http://arxiv.org/pdf/2504.15047v1)

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at https://github.com/knoveleng/rainbowplus, supporting reproducibility and future research in LLM red-teaming.

摘要: 大型语言模型（LLM）表现出非凡的能力，但很容易受到对抗提示的影响，这些提示利用漏洞产生不安全或有偏见的输出。现有的红色团队方法通常面临可扩展性挑战、资源密集型要求或攻击策略的多样性有限。我们提出RainbowPlus，这是一种植根于进化计算的新型红色团队框架，通过自适应质量多样性（QD）搜索来增强对抗提示生成，该搜索扩展了MAP-Elites等经典进化算法，并为语言模型量身定制的创新。通过采用多元素档案来存储各种高质量提示，并采用全面的适应度函数来同时评估多个提示，RainbowPlus克服了Rainbow Teaming等现有QD方法中单提示档案和成对比较的限制。在六个基准数据集和四个开源LLM上比较RainbowPlus与QD方法的实验证明了卓越的攻击成功率（ASB）和多样性（Diverse-Score $\大约0.84$），生成高达100倍的独特提示（例如，Ministral-8B-Direct-2410为10，418 vs 100）。与HarmBench数据集中的九种最先进方法（12个LLM（10个开源，2个封闭源）相比，RainbowPlus的平均ASB为81.1%，超过AutoDAN-Turbo 3.9%，速度快9倍（1.45小时vs 13.50小时）。我们的开源实施促进了LLM安全性的进一步进步，为漏洞评估提供了可扩展的工具。代码和资源可在https：//github.com/knoveleng/rainbowplus上公开获取，支持LLM红色团队的再现性和未来研究。



## **43. An Information-theoretic Security Analysis of Honeyword**

蜜语的信息论安全分析 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2311.10960v2) [paper-pdf](http://arxiv.org/pdf/2311.10960v2)

**Authors**: Pengcheng Su, Haibo Cheng, Wenting Li, Ping Wang

**Abstract**: Honeyword is a representative "honey" technique that employs decoy objects to mislead adversaries and protect the real ones. To assess the security of a Honeyword system, two metrics--flatness and success-number--have been proposed and evaluated using various simulated attackers. Existing evaluations typically apply statistical learning methods to distinguish real passwords from decoys on real-world datasets. However, such evaluations may overestimate the system's security, as more effective distinguishing attacks could potentially exist.   In this paper, we aim to analyze the security of Honeyword systems under the strongest theoretical attack, rather than relying on specific, expert-crafted attacks evaluated in prior experimental studies. We first derive mathematical expressions for the flatness and success-number under the strongest attack. We conduct analyses and computations for several typical scenarios, and determine the security of honeyword generation methods using a uniform distribution and the List model as examples.   We further evaluate the security of existing honeyword generation methods based on password probability models (PPMs), which depends on the sample size used for training. We investigate, for the first time, the sample complexity of several representative PPMs, introducing two novel polynomial-time approximation schemes for computing the total variation between PCFG models and between higher-order Markov models. Our experimental results show that for small-scale password distributions, sample sizes on the order of millions--often tens of millions--are required to reduce the total variation below 0.1. A surprising result is that we establish an equivalence between flatness and total variation, thus bridging the theoretical study of Honeyword systems with classical information theory. Finally, we discuss the practical implications of our findings.

摘要: 蜜语是一种典型的“蜂蜜”技术，它利用诱饵对象来误导攻击者并保护真实对象。为了评估一个Honeyword系统的安全性，两个指标-平坦度和成功数-已被提出，并使用各种模拟攻击者进行评估。现有的评估通常应用统计学习方法来区分真实世界数据集上的真实密码和诱饵。然而，这种评估可能会高估系统的安全性，因为可能存在更有效的区分攻击。   在本文中，我们的目标是分析最强理论攻击下的Honeyword系统的安全性，而不是依赖于在先前的实验研究中评估的特定的、专家制作的攻击。我们首先推导出最强攻击下的平坦性和成功数的数学表达式。针对几种典型的情况进行了分析和计算，并以均匀分布和List模型为例，确定了蜜词生成方法的安全性。   我们进一步评估现有的基于密码概率模型（PPM），这取决于用于训练的样本大小的蜜词生成方法的安全性。我们首次研究了几种有代表性的PPP的样本复杂性，引入了两种新型的多项时间逼近方案来计算PCFG模型之间和更高次Markov模型之间的总方差。我们的实验结果表明，对于小规模密码分发，需要数百万（通常是数千万）数量级的样本量才能将总变异减少到0.1以下。一个令人惊讶的结果是，我们建立了平坦性和总变异性之间的等效性，从而将蜜语系统的理论研究与经典信息论联系起来。最后，我们讨论了我们发现的实际影响。



## **44. Transferable Adversarial Attacks on SAM and Its Downstream Models**

对Sam及其下游模型的可转移对抗攻击 cs.LG

update fig 1

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.20197v3) [paper-pdf](http://arxiv.org/pdf/2410.20197v3)

**Authors**: Song Xia, Wenhan Yang, Yi Yu, Xun Lin, Henghui Ding, Ling-Yu Duan, Xudong Jiang

**Abstract**: The utilization of large foundational models has a dilemma: while fine-tuning downstream tasks from them holds promise for making use of the well-generalized knowledge in practical applications, their open accessibility also poses threats of adverse usage. This paper, for the first time, explores the feasibility of adversarial attacking various downstream models fine-tuned from the segment anything model (SAM), by solely utilizing the information from the open-sourced SAM. In contrast to prevailing transfer-based adversarial attacks, we demonstrate the existence of adversarial dangers even without accessing the downstream task and dataset to train a similar surrogate model. To enhance the effectiveness of the adversarial attack towards models fine-tuned on unknown datasets, we propose a universal meta-initialization (UMI) algorithm to extract the intrinsic vulnerability inherent in the foundation model, which is then utilized as the prior knowledge to guide the generation of adversarial perturbations. Moreover, by formulating the gradient difference in the attacking process between the open-sourced SAM and its fine-tuned downstream models, we theoretically demonstrate that a deviation occurs in the adversarial update direction by directly maximizing the distance of encoded feature embeddings in the open-sourced SAM. Consequently, we propose a gradient robust loss that simulates the associated uncertainty with gradient-based noise augmentation to enhance the robustness of generated adversarial examples (AEs) towards this deviation, thus improving the transferability. Extensive experiments demonstrate the effectiveness of the proposed universal meta-initialized and gradient robust adversarial attack (UMI-GRAT) toward SAMs and their downstream models. Code is available at https://github.com/xiasong0501/GRAT.

摘要: 大型基础模型的利用面临一个困境：虽然对它们的下游任务进行微调有望在实际应用中利用良好的概括性知识，但它们的开放可访问性也构成了不良使用的威胁。本文首次探讨了对抗攻击从细分任何模型（Sam）微调的各种下游模型的可行性，仅利用开源Sam的信息。与流行的基于传输的对抗攻击相反，我们证明了对抗危险的存在，即使不访问下游任务和数据集来训练类似的代理模型。为了提高针对在未知数据集上微调的模型的对抗攻击的有效性，我们提出了一种通用元初始化（UMI）算法来提取基础模型中固有的内在漏洞，然后将其用作先验知识来指导对抗性扰动的生成。此外，通过公式化开放源代码的Sam及其微调下游模型之间攻击过程中的梯度差异，我们从理论上证明了通过直接最大化开放源代码的特征嵌入的距离，对抗性更新方向上会发生偏差。因此，我们提出了一种梯度鲁棒性损失，通过基于梯度的噪音增强来模拟相关的不确定性，以增强生成的对抗性示例（AE）对该偏差的鲁棒性，从而提高可移植性。大量实验证明了所提出的通用元初始化和梯度鲁棒对抗攻击（UMI-GRAT）对自组装体及其下游模型的有效性。代码可在https://github.com/xiasong0501/GRAT上获取。



## **45. PA-Boot: A Formally Verified Authentication Protocol for Multiprocessor Secure Boot**

PA-Boot：用于多处理器安全引导的正式验证认证协议 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2209.07936v3) [paper-pdf](http://arxiv.org/pdf/2209.07936v3)

**Authors**: Zhuoruo Zhang, Rui Chang, Mingshuai Chen, Wenbo Shen, Chenyang Yu, He Huang, Qinming Dai, Yongwang Zhao

**Abstract**: Hardware supply-chain attacks are raising significant security threats to the boot process of multiprocessor systems. This paper identifies a new, prevalent hardware supply-chain attack surface that can bypass multiprocessor secure boot due to the absence of processor-authentication mechanisms. To defend against such attacks, we present PA-Boot, the first formally verified processor-authentication protocol for secure boot in multiprocessor systems. PA-Boot is proved functionally correct and is guaranteed to detect multiple adversarial behaviors, e.g., processor replacements, man-in-the-middle attacks, and tampering with certificates. The fine-grained formalization of PA-Boot and its fully mechanized security proofs are carried out in the Isabelle/HOL theorem prover with 306 lemmas/theorems and ~7,100 LoC. Experiments on a proof-of-concept implementation indicate that PA-Boot can effectively identify boot-process attacks with a considerably minor overhead and thereby improve the security of multiprocessor systems.

摘要: 硬件供应链攻击正在对多处理器系统的引导过程构成重大安全威胁。本文指出了一种新的、流行的硬件供应链攻击面，由于缺乏处理器身份验证机制，该攻击面可以绕过多处理器的安全引导。为了抵御此类攻击，我们提出了PA-Boot，这是第一个经过正式验证的处理器身份验证协议，用于多处理器系统中的安全引导。PA-Boot被证明功能正确，并保证能够检测到多种对抗行为，例如，处理器更换、中间人攻击和篡改证书。PA-Boot的细粒度形式化及其完全机械化的安全证明是在Isabelle/HOL定理证明器中执行的，具有306个引理/定理和约7，100 LoC。概念验证实现的实验表明，PA-Boot可以以相当小的负担有效地识别引导进程攻击，从而提高多处理器系统的安全性。



## **46. Verifying Robust Unlearning: Probing Residual Knowledge in Unlearned Models**

稳健的非学习：探索未学习模型中的剩余知识 cs.LG

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14798v1) [paper-pdf](http://arxiv.org/pdf/2504.14798v1)

**Authors**: Hao Xuan, Xingyu Li

**Abstract**: Machine Unlearning (MUL) is crucial for privacy protection and content regulation, yet recent studies reveal that traces of forgotten information persist in unlearned models, enabling adversaries to resurface removed knowledge. Existing verification methods only confirm whether unlearning was executed, failing to detect such residual information leaks. To address this, we introduce the concept of Robust Unlearning, ensuring models are indistinguishable from retraining and resistant to adversarial recovery. To empirically evaluate whether unlearning techniques meet this security standard, we propose the Unlearning Mapping Attack (UMA), a post-unlearning verification framework that actively probes models for forgotten traces using adversarial queries. Extensive experiments on discriminative and generative tasks show that existing unlearning techniques remain vulnerable, even when passing existing verification metrics. By establishing UMA as a practical verification tool, this study sets a new standard for assessing and enhancing machine unlearning security.

摘要: 机器非学习（MUL）对于隐私保护和内容监管至关重要，但最近的研究表明，被遗忘的信息的痕迹仍然存在于未学习的模型中，使对手能够重新暴露被删除的知识。现有的验证方法仅确认是否执行了取消学习，无法检测到此类剩余信息泄露。为了解决这个问题，我们引入了稳健的非学习概念，确保模型与再培训没有区别，并且能够抵抗对抗性恢复。为了从经验上评估取消学习技术是否满足这一安全标准，我们提出了取消学习映射攻击（UMA），这是一个取消学习后验证框架，它使用对抗性查询主动探测模型中的遗忘痕迹。关于区分性和生成性任务的广泛实验表明，即使通过了现有的验证指标，现有的取消学习技术仍然很脆弱。通过将UMA建立为实用验证工具，本研究为评估和增强机器学习安全性设定了新标准。



## **47. Modality Unified Attack for Omni-Modality Person Re-Identification**

全方位人重新识别的情态统一攻击 cs.CV

9 pages,3 figures

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2501.12761v2) [paper-pdf](http://arxiv.org/pdf/2501.12761v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yunfeng Ma, Yaonan Wang

**Abstract**: Deep learning based person re-identification (re-id) models have been widely employed in surveillance systems. Recent studies have demonstrated that black-box single-modality and cross-modality re-id models are vulnerable to adversarial examples (AEs), leaving the robustness of multi-modality re-id models unexplored. Due to the lack of knowledge about the specific type of model deployed in the target black-box surveillance system, we aim to generate modality unified AEs for omni-modality (single-, cross- and multi-modality) re-id models. Specifically, we propose a novel Modality Unified Attack method to train modality-specific adversarial generators to generate AEs that effectively attack different omni-modality models. A multi-modality model is adopted as the surrogate model, wherein the features of each modality are perturbed by metric disruption loss before fusion. To collapse the common features of omni-modality models, Cross Modality Simulated Disruption approach is introduced to mimic the cross-modality feature embeddings by intentionally feeding images to non-corresponding modality-specific subnetworks of the surrogate model. Moreover, Multi Modality Collaborative Disruption strategy is devised to facilitate the attacker to comprehensively corrupt the informative content of person images by leveraging a multi modality feature collaborative metric disruption loss. Extensive experiments show that our MUA method can effectively attack the omni-modality re-id models, achieving 55.9%, 24.4%, 49.0% and 62.7% mean mAP Drop Rate, respectively.

摘要: 基于深度学习的人员重新识别（re-id）模型已广泛应用于监控系统中。最近的研究表明，黑匣子单模式和跨模式re-id模型容易受到对抗性示例（AE）的影响，因此多模式re-id模型的稳健性尚未得到探索。由于缺乏对目标黑匣子监控系统中部署的特定类型模型的了解，我们的目标是为全模式（单模式、交叉模式和多模式）re-id模型生成模式统一AE。具体来说，我们提出了一种新颖的模式统一攻击方法来训练特定于模式的对抗生成器，以生成有效攻击不同全模式模型的AE。采用多模式模型作为代理模型，其中每个模式的特征在融合之前受到度量中断损失的干扰。为了瓦解全模式模型的共同特征，引入了跨模式模拟破坏方法，通过有意地将图像提供给代理模型的非对应模式特定子网络来模拟跨模式特征嵌入。此外，多模式协同破坏策略旨在帮助攻击者通过利用多模式特征协同指标破坏损失来全面破坏人物图像的信息内容。大量实验表明，我们的MUA方法可以有效攻击全模式re-id模型，平均mAP下降率分别达到55.9%、24.4%、49.0%和62.7%。



## **48. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

展望未来：通过揭露对抗性合同来防止DeFi攻击 cs.CR

23 pages, 7 figures; Accepted to FSE 2025, to be published in The  Proceedings of the ACM on Software Engineering (PACMSE)

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2401.07261v6) [paper-pdf](http://arxiv.org/pdf/2401.07261v6)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: The exploitation of smart contract vulnerabilities in Decentralized Finance (DeFi) has resulted in financial losses exceeding 3 billion US dollars. Existing defense mechanisms primarily focus on detecting and reacting to adversarial transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively. Based on the fact that most attack logic rely on deploying intermediate smart contracts as supporting components to the exploitation of victim contracts, novel detection methods have been proposed that focus on identifying these adversarial contracts instead of adversarial transactions. However, previous state-of-the-art approaches in this direction have failed to produce results satisfactory enough for real-world deployment. In this paper, we propose LookAhead, a new framework for detecting DeFi attacks via unveiling adversarial contracts. LookAhead leverages common attack patterns, code semantics and intrinsic characteristics found in adversarial smart contracts to train Machine Learning (ML)-based classifiers that can effectively distinguish adversarial contracts from benign ones and make timely predictions of different types of potential attacks. Experiments on our labeled datasets show that LookAhead achieves an F1-score as high as 0.8966, which represents an improvement of over 44.4% compared to the previous state-of-the-art solution, with a False Positive Rate (FPR) at only 0.16%.

摘要: 利用去中心金融（DeFi）中的智能合约漏洞已造成超过30亿美元的财务损失。现有的防御机制主要专注于检测和反应攻击者针对受害者合同执行的对抗交易。然而，随着私有交易池的出现，交易直接发送给矿工，而无需首先出现在公共内存池中，当前的检测工具在有效识别攻击活动方面面临着重大挑战。基于大多数攻击逻辑依赖于部署中间智能合同作为利用受害者合同的支持组件这一事实，人们提出了新型检测方法，重点关注识别这些对抗性合同而不是对抗性交易。然而，之前在此方向上的最新方法未能产生足够令人满意的结果，以满足现实世界的部署。在本文中，我们提出了LookAhead，这是一个通过公布对抗性合同来检测DeFi攻击的新框架。LookAhead利用对抗性智能合同中的常见攻击模式、代码语义和内在特征来训练基于机器学习（ML）的分类器，可以有效区分对抗性合同与良性合同，并对不同类型的潜在攻击做出及时预测。对我们标记的数据集的实验表明，LookAhead的F1评分高达0.8966，与之前的最先进解决方案相比提高了44.4%以上，假阳性率（FPR）仅为0.16%。



## **49. Human-AI Collaboration in Cloud Security: Cognitive Hierarchy-Driven Deep Reinforcement Learning**

云安全中的人机协作：认知层次驱动的深度强化学习 cs.CR

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2502.16054v2) [paper-pdf](http://arxiv.org/pdf/2502.16054v2)

**Authors**: Zahra Aref, Sheng Wei, Narayan B. Mandayam

**Abstract**: Given the complexity of multi-tenant cloud environments and the growing need for real-time threat mitigation, Security Operations Centers (SOCs) must adopt AI-driven adaptive defense mechanisms to counter Advanced Persistent Threats (APTs). However, SOC analysts face challenges in handling adaptive adversarial tactics, requiring intelligent decision-support frameworks. We propose a Cognitive Hierarchy Theory-driven Deep Q-Network (CHT-DQN) framework that models interactive decision-making between SOC analysts and AI-driven APT bots. The SOC analyst (defender) operates at cognitive level-1, anticipating attacker strategies, while the APT bot (attacker) follows a level-0 policy. By incorporating CHT into DQN, our framework enhances adaptive SOC defense using Attack Graph (AG)-based reinforcement learning. Simulation experiments across varying AG complexities show that CHT-DQN consistently achieves higher data protection and lower action discrepancies compared to standard DQN. A theoretical lower bound further confirms its superiority as AG complexity increases. A human-in-the-loop (HITL) evaluation on Amazon Mechanical Turk (MTurk) reveals that SOC analysts using CHT-DQN-derived transition probabilities align more closely with adaptive attackers, leading to better defense outcomes. Moreover, human behavior aligns with Prospect Theory (PT) and Cumulative Prospect Theory (CPT): participants are less likely to reselect failed actions and more likely to persist with successful ones. This asymmetry reflects amplified loss sensitivity and biased probability weighting -- underestimating gains after failure and overestimating continued success. Our findings highlight the potential of integrating cognitive models into deep reinforcement learning to improve real-time SOC decision-making for cloud security.

摘要: 鉴于多租户云环境的复杂性以及对实时威胁缓解的日益增长的需求，安全运营中心（SOC）必须采用人工智能驱动的自适应防御机制来应对高级持续性威胁（APT）。然而，SOC分析师在处理自适应对抗策略方面面临挑战，需要智能的决策支持框架。我们提出了一个认知层次理论驱动的深度Q网络（CHT-DQN）框架，该框架对SOC分析师和人工智能驱动的APT机器人之间的交互决策进行建模。SOC分析师（防御者）在认知级别1下工作，预测攻击者策略，而APT机器人（攻击者）遵循0级策略。通过将CHT整合到DQN中，我们的框架使用基于攻击图（AG）的强化学习来增强自适应SOC防御。不同AG复杂性的模拟实验表明，与标准DQN相比，CHT-DQN始终实现更高的数据保护和更低的动作差异。随着AG复杂性的增加，理论下限进一步证实了其优势。对Amazon Mechanical Turk（MTurk）进行的人在环（HITL）评估显示，使用CHT-DQN推导的转移概率的SOC分析师与自适应攻击者更紧密地一致，从而获得更好的防御结果。此外，人类行为与前景理论（PT）和累积前景理论（CPD）一致：参与者不太可能重新选择失败的行动，而更有可能坚持成功的行动。这种不对称反映了损失敏感性的放大和概率加权的偏差--低估了失败后的收益并高估了持续成功。我们的研究结果强调了将认知模型集成到深度强化学习中以改善云安全的实时SOC决策的潜力。



## **50. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

大型语言模型作为软件分析中稳健的数据生成器：我们已经做到了吗？ cs.SE

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2411.10565v2) [paper-pdf](http://arxiv.org/pdf/2411.10565v2)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.

摘要: 大型语言模型（LLM）生成的数据越来越多地用于软件分析，但目前尚不清楚该数据与人类编写的数据相比如何，特别是当模型暴露于对抗场景时。对抗性攻击可能会损害软件系统的可靠性和安全性，因此，与作为模型性能基准的人类编写数据相比，了解LLM生成的数据在这些条件下的表现如何，可以为LLM生成的数据是否提供类似的稳健性和有效性提供有价值的见解。为了解决这一差距，我们系统地评估和比较人类编写的数据和LLM生成的数据的质量，以便在对抗性攻击的背景下微调稳健的预训练模型（Ptms）。我们评估了六种广泛使用的PtM的稳健性，这些PtM在对抗性攻击之前和之后根据人类编写和LLM生成的数据进行了微调。该评估在三个流行的软件分析任务中使用了九种最先进的（SOTA）对抗性攻击技术：克隆检测，代码摘要和代码审查讨论中的情感分析。此外，我们使用11个相似性度量来分析生成的对抗性示例的质量。我们的研究结果表明，虽然对LLM生成的数据进行微调的PTM与对人类编写的数据进行微调的PTM具有竞争力，但它们在软件分析任务中对对抗性攻击的鲁棒性较低。我们的研究强调了进一步探索提高LLM生成的训练数据质量的必要性，以开发高性能且能够抵御软件分析中的对抗性攻击的模型。



