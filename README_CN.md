# Latest Adversarial Attack Papers
**update at 2025-12-19 15:45:31**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. PrivateXR: Defending Privacy Attacks in Extended Reality Through Explainable AI-Guided Differential Privacy**

PrivateXR：通过可解释的人工智能引导的差异隐私在延展实境中防御隐私攻击 cs.CR

Published in the IEEE ISMAR 2025 conference

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16851v1) [paper-pdf](https://arxiv.org/pdf/2512.16851v1)

**Authors**: Ripan Kumar Kundu, Istiak Ahmed, Khaza Anuarul Hoque

**Abstract**: The convergence of artificial AI and XR technologies (AI XR) promises innovative applications across many domains. However, the sensitive nature of data (e.g., eye-tracking) used in these systems raises significant privacy concerns, as adversaries can exploit these data and models to infer and leak personal information through membership inference attacks (MIA) and re-identification (RDA) with a high success rate. Researchers have proposed various techniques to mitigate such privacy attacks, including differential privacy (DP). However, AI XR datasets often contain numerous features, and applying DP uniformly can introduce unnecessary noise to less relevant features, degrade model accuracy, and increase inference time, limiting real-time XR deployment. Motivated by this, we propose a novel framework combining explainable AI (XAI) and DP-enabled privacy-preserving mechanisms to defend against privacy attacks. Specifically, we leverage post-hoc explanations to identify the most influential features in AI XR models and selectively apply DP to those features during inference. We evaluate our XAI-guided DP approach on three state-of-the-art AI XR models and three datasets: cybersickness, emotion, and activity classification. Our results show that the proposed method reduces MIA and RDA success rates by up to 43% and 39%, respectively, for cybersickness tasks while preserving model utility with up to 97% accuracy using Transformer models. Furthermore, it improves inference time by up to ~2x compared to traditional DP approaches. To demonstrate practicality, we deploy the XAI-guided DP AI XR models on an HTC VIVE Pro headset and develop a user interface (UI), namely PrivateXR, allowing users to adjust privacy levels (e.g., low, medium, high) while receiving real-time task predictions, protecting user privacy during XR gameplay.

摘要: 人工AI和XR技术（AI XR）的融合有望在许多领域实现创新应用。然而，数据的敏感性（例如，这些系统中使用的眼睛跟踪）引发了严重的隐私问题，因为对手可以利用这些数据和模型通过成员资格推断攻击（MIA）和重新识别（RDA）来推断和泄露个人信息，成功率很高。研究人员提出了各种技术来缓解此类隐私攻击，包括差异隐私（DP）。然而，AI XR数据集通常包含大量特征，统一应用DP可能会给不太相关的特征带来不必要的噪音，降低模型准确性，并增加推理时间，从而限制实时XR部署。出于此动机，我们提出了一种新颖的框架，将可解释人工智能（XAI）和DP支持的隐私保护机制相结合，以抵御隐私攻击。具体来说，我们利用事后解释来识别AI XR模型中最有影响力的特征，并在推理期间选择性地将DP应用于这些特征。我们在三个最先进的AI XR模型和三个数据集上评估了我们的XAI引导的DP方法：网络疾病、情绪和活动分类。我们的结果表明，对于网瘾任务，所提出的方法将MIA和RDA成功率分别降低了高达43%和39%，同时使用Transformer模型保留了高达97%的准确性的模型效用。此外，与传统DP方法相比，它将推理时间缩短了约2倍。为了展示实用性，我们在HTC VIVE Pro耳机上部署了XAI引导的DP AI XR模型，并开发了用户界面（UI），即PrivateXR，允许用户调整隐私级别（例如，低、中、高），同时接收实时任务预测，在XR游戏过程中保护用户隐私。



## **2. Misspecified Crame-Rao Bound for AoA Estimation at a ULA under a Spoofing Attack**

欺骗攻击下的预设AoA估计的Crame-Rao界被错误指定 eess.SP

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16735v1) [paper-pdf](https://arxiv.org/pdf/2512.16735v1)

**Authors**: Sotiris Skaperas, Arsenia Chorti

**Abstract**: A framework is presented for analyzing the impact of active attacks to location-based physical layer authentication (PLA) using the machinery of misspecified Cramér--Rao bound (MCRB). In this work, we focus on the MCRB in the angle-of-arrival (AoA) based authentication of a single antenna user when the verifier posseses an $M$ antenna element uniform linear array (ULA), assuming deterministic pilot signals; in our system model the presence of a spoofing adversary with an arbitrary number $L$ of antenna elements is assumed. We obtain a closed-form expression for the MCRB and demonstrate that the attack introduces in it a penalty term compared to the classic CRB, which does not depend on the signal-to-noise ratio (SNR) but on the adversary's location, the array geometry and the attacker precoding vector.

摘要: 提出了一个框架，用于分析主动攻击对使用错误指定的Cramér-Rao界（MCRB）机制的基于位置的物理层认证（PLA）的影响。在这项工作中，我们重点关注当验证者拥有$M$天线元件均匀线性阵列（RST）时，单天线用户基于到达角（AoA）的认证中的MCRB，假设导频信号确定性;在我们的系统模型中，假设存在具有任意数量$L$的欺骗对手。我们获得了MCRB的封闭形式表达，并证明与经典CRB相比，攻击在其中引入了一个罚项，该罚项不取决于信噪比（SNR），而是取决于对手的位置、阵列几何形状和攻击者预编码载体。



## **3. Dual-View Inference Attack: Machine Unlearning Amplifies Privacy Exposure**

双视图推理攻击：机器遗忘放大隐私暴露 cs.LG

Accepeted by AAAI2026

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16126v1) [paper-pdf](https://arxiv.org/pdf/2512.16126v1)

**Authors**: Lulu Xue, Shengshan Hu, Linqiang Qian, Peijin Guo, Yechao Zhang, Minghui Li, Yanjun Zhang, Dayong Ye, Leo Yu Zhang

**Abstract**: Machine unlearning is a newly popularized technique for removing specific training data from a trained model, enabling it to comply with data deletion requests. While it protects the rights of users requesting unlearning, it also introduces new privacy risks. Prior works have primarily focused on the privacy of data that has been unlearned, while the risks to retained data remain largely unexplored. To address this gap, we focus on the privacy risks of retained data and, for the first time, reveal the vulnerabilities introduced by machine unlearning under the dual-view setting, where an adversary can query both the original and the unlearned models. From an information-theoretic perspective, we introduce the concept of {privacy knowledge gain} and demonstrate that the dual-view setting allows adversaries to obtain more information than querying either model alone, thereby amplifying privacy leakage. To effectively demonstrate this threat, we propose DVIA, a Dual-View Inference Attack, which extracts membership information on retained data using black-box queries to both models. DVIA eliminates the need to train an attack model and employs a lightweight likelihood ratio inference module for efficient inference. Experiments across different datasets and model architectures validate the effectiveness of DVIA and highlight the privacy risks inherent in the dual-view setting.

摘要: 机器取消学习是一种新流行的技术，用于从训练后的模型中删除特定的训练数据，使其能够遵守数据删除请求。虽然它保护了请求取消学习的用户的权利，但它也带来了新的隐私风险。之前的作品主要关注未了解的数据的隐私，而保留数据的风险在很大程度上仍未被探索。为了解决这一差距，我们重点关注保留数据的隐私风险，并首次揭示了双视图环境下机器取消学习引入的漏洞，其中对手可以查询原始和未学习的模型。从信息论的角度，我们引入了{隐私知识获得}的概念，并证明双视角设置允许对手获得比单独查询任何一个模型更多的信息，从而放大了隐私泄露。为了有效地证明这种威胁，我们提出了DVIA，这是一种双视图推理攻击，它使用对两个模型的黑匣子查询来提取保留数据的成员信息。DVIA消除了训练攻击模型的需要，并采用轻量级似然比推理模块进行高效推理。跨不同数据集和模型架构的实验验证了DVIA的有效性，并强调了双视图设置中固有的隐私风险。



## **4. Autoencoder-based Denoising Defense against Adversarial Attacks on Object Detection**

基于自动编码器的目标检测对抗攻击去噪防御 cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16123v1) [paper-pdf](https://arxiv.org/pdf/2512.16123v1)

**Authors**: Min Geun Song, Gang Min Kim, Woonmin Kim, Yongsik Kim, Jeonghyun Sim, Sangbeom Park, Huy Kang Kim

**Abstract**: Deep learning-based object detection models play a critical role in real-world applications such as autonomous driving and security surveillance systems, yet they remain vulnerable to adversarial examples. In this work, we propose an autoencoder-based denoising defense to recover object detection performance degraded by adversarial perturbations. We conduct adversarial attacks using Perlin noise on vehicle-related images from the COCO dataset, apply a single-layer convolutional autoencoder to remove the perturbations, and evaluate detection performance using YOLOv5. Our experiments demonstrate that adversarial attacks reduce bbox mAP from 0.2890 to 0.1640, representing a 43.3% performance degradation. After applying the proposed autoencoder defense, bbox mAP improves to 0.1700 (3.7% recovery) and bbox mAP@50 increases from 0.2780 to 0.3080 (10.8% improvement). These results indicate that autoencoder-based denoising can provide partial defense against adversarial attacks without requiring model retraining.

摘要: 基于深度学习的对象检测模型在自动驾驶和安全监控系统等现实世界应用中发挥着至关重要的作用，但它们仍然容易受到对抗示例的影响。在这项工作中，我们提出了一种基于自动编码器的去噪防御，以恢复因对抗性扰动而降低的对象检测性能。我们使用Perlin噪音对COCO数据集的车辆相关图像进行对抗攻击，应用单层卷积自动编码器来消除扰动，并使用YOLOv 5评估检测性能。我们的实验表明，对抗性攻击将bbox mAP从0.2890降低到0.1640，代表43.3%的性能下降。应用拟议的自动编码器防御后，bbox mAP提高到0.1700（恢复率3.7%），bbox mAP@50从0.2780提高到0.3080（改善10.8%）。这些结果表明，基于自动编码器的去噪可以在不需要模型重新训练的情况下提供部分防御对抗攻击。



## **5. From Risk to Resilience: Towards Assessing and Mitigating the Risk of Data Reconstruction Attacks in Federated Learning**

从风险到韧性：评估和缓解联邦学习中数据重建攻击的风险 cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15460v1) [paper-pdf](https://arxiv.org/pdf/2512.15460v1)

**Authors**: Xiangrui Xu, Zhize Li, Yufei Han, Bin Wang, Jiqiang Liu, Wei Wang

**Abstract**: Data Reconstruction Attacks (DRA) pose a significant threat to Federated Learning (FL) systems by enabling adversaries to infer sensitive training data from local clients. Despite extensive research, the question of how to characterize and assess the risk of DRAs in FL systems remains unresolved due to the lack of a theoretically-grounded risk quantification framework. In this work, we address this gap by introducing Invertibility Loss (InvLoss) to quantify the maximum achievable effectiveness of DRAs for a given data instance and FL model. We derive a tight and computable upper bound for InvLoss and explore its implications from three perspectives. First, we show that DRA risk is governed by the spectral properties of the Jacobian matrix of exchanged model updates or feature embeddings, providing a unified explanation for the effectiveness of defense methods. Second, we develop InvRE, an InvLoss-based DRA risk estimator that offers attack method-agnostic, comprehensive risk evaluation across data instances and model architectures. Third, we propose two adaptive noise perturbation defenses that enhance FL privacy without harming classification accuracy. Extensive experiments on real-world datasets validate our framework, demonstrating its potential for systematic DRA risk evaluation and mitigation in FL systems.

摘要: 数据重建攻击（NPS）使对手能够从本地客户端推断敏感训练数据，从而对联邦学习（FL）系统构成重大威胁。尽管进行了广泛的研究，但由于缺乏基于理论的风险量化框架，如何描述和评估FL系统中DSA风险的问题仍未得到解决。在这项工作中，我们通过引入可逆损失（InvLoss）来量化给定数据实例和FL模型的DSA的最大可实现有效性来解决这一差距。我们推导出InvLoss的紧密且可计算的上限，并从三个角度探讨其含义。首先，我们表明，数据库风险由交换模型更新或特征嵌入的雅可比矩阵的谱属性决定，为防御方法的有效性提供了统一的解释。其次，我们开发InvRE，这是一种基于InvLoss的Inver-risk估计器，它提供跨数据实例和模型架构的攻击方法不可知的全面风险评估。第三，我们提出了两种自适应噪音扰动防御措施，可以在不损害分类准确性的情况下增强FL隐私。对现实世界数据集的广泛实验验证了我们的框架，展示了其在FL系统中进行系统性NPS风险评估和缓解的潜力。



## **6. Talking to the Airgap: Exploiting Radio-Less Embedded Devices as Radio Receivers**

与Airgap交谈：利用无线电嵌入式设备作为无线电接收器 cs.CR

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15387v1) [paper-pdf](https://arxiv.org/pdf/2512.15387v1)

**Authors**: Paul Staat, Daniel Davidovich, Christof Paar

**Abstract**: Intelligent electronics are deeply embedded in critical infrastructures and must remain reliable, particularly against deliberate attacks. To minimize risks and impede remote compromise, sensitive systems can be physically isolated from external networks, forming an airgap. Yet, airgaps can still be infiltrated by capable adversaries gaining code execution. Prior research has shown that attackers can then attempt to wirelessly exfiltrate data across the airgap by exploiting unintended radio emissions. In this work, we demonstrate reversal of this link: malicious code execution on embedded devices can enable wireless infiltration of airgapped systems without any hardware modification. In contrast to previous infiltration methods that depend on dedicated sensors (e.g., microphones, LEDs, or temperature sensors) or require strict line-of-sight, we show that unmodified, sensor-less embedded devices can inadvertently act as radio receivers. This phenomenon stems from parasitic RF sensitivity in PCB traces and on-chip analog-to-digital converters (ADCs), allowing external transmissions to be received and decoded entirely in software.   Across twelve commercially available embedded devices and two custom prototypes, we observe repeatable reception in the 300-1000 MHz range, with detectable signal power as low as 1 mW. To this end, we propose a systematic methodology to identify device configurations that foster such radio sensitivities and comprehensively evaluate their feasibility for wireless data reception. Exploiting these sensitivities, we demonstrate successful data reception over tens of meters, even in non-line-of-sight conditions and show that the reception sensitivities accommodate data rates of up to 100 kbps. Our findings reveal a previously unexplored command-and-control vector for air-gapped systems while challenging assumptions about their inherent isolation. [shortened]

摘要: 智能电子产品深深嵌入关键基础设施中，必须保持可靠性，特别是针对故意攻击。为了最大限度地减少风险并阻止远程破坏，敏感系统可以与外部网络物理隔离，形成气间隙。然而，漏洞仍然可能被有能力的对手渗透，以获得代码执行。之前的研究表明，攻击者可以尝试通过利用无意的无线电发射来通过空气间隙无线传输数据。在这项工作中，我们展示了这个链接的逆转：嵌入式设备上的恶意代码执行可以在不修改任何硬件的情况下实现对空间隙系统的无线渗透。与之前依赖于专用传感器的渗透方法相反（例如，麦克风、LED或温度传感器）或需要严格的视线，我们表明未经修改的无传感器嵌入式设备可能会无意中充当无线电接收器。这种现象源于PCB走线和片内模拟数字转换器（ADC）中的寄生RF灵敏度，允许外部传输完全通过软件接收和解码。   在十二种市售嵌入式设备和两种定制原型中，我们观察到300-1000 MHz范围内的可重复接收，可检测信号功率低至1 MW。为此，我们提出了一种系统性的方法来识别能够促进此类无线电敏感性的设备配置，并全面评估其无线数据接收的可行性。利用这些灵敏度，我们展示了即使在非视线条件下也能在数十米范围内成功接收数据，并表明接收灵敏度可适应高达100 kMbps的数据速率。我们的研究结果揭示了一种以前未探索的气间隙系统的命令和控制载体，同时挑战了有关其固有隔离性的假设。[缩短]



## **7. Bounty Hunter: Autonomous, Comprehensive Emulation of Multi-Faceted Adversaries**

赏金猎人：多面对手的自主、全面模拟 cs.CR

15 pages, 9 figures

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15275v1) [paper-pdf](https://arxiv.org/pdf/2512.15275v1)

**Authors**: Louis Hackländer-Jansen, Rafael Uetz, Martin Henze

**Abstract**: Adversary emulation is an essential procedure for cybersecurity assessments such as evaluating an organization's security posture or facilitating structured training and research in dedicated environments. To allow for systematic and time-efficient assessments, several approaches from academia and industry have worked towards the automation of adversarial actions. However, they exhibit significant limitations regarding autonomy, tactics coverage, and real-world applicability. Consequently, adversary emulation remains a predominantly manual task requiring substantial human effort and security expertise - even amidst the rise of Large Language Models. In this paper, we present Bounty Hunter, an automated adversary emulation method, designed and implemented as an open-source plugin for the popular adversary emulation platform Caldera, that enables autonomous emulation of adversaries with multi-faceted behavior while providing a wide coverage of tactics. To this end, it realizes diverse adversarial behavior, such as different levels of detectability and varying attack paths across repeated emulations. By autonomously compromising a simulated enterprise network, Bounty Hunter showcases its ability to achieve given objectives without prior knowledge of its target, including pre-compromise, initial compromise, and post-compromise attack tactics. Overall, Bounty Hunter facilitates autonomous, comprehensive, and multi-faceted adversary emulation to help researchers and practitioners in performing realistic and time-efficient security assessments, training exercises, and intrusion detection research.

摘要: Adobile仿真是网络安全评估的重要程序，例如评估组织的安全态势或促进专用环境中的结构化培训和研究。为了进行系统性且高效的评估，学术界和工业界的多种方法致力于对抗行动的自动化。然而，它们在自主性、战术覆盖范围和现实世界的适用性方面表现出显着的局限性。因此，对手模拟仍然是一项主要的手动任务，需要大量的人力和安全专业知识--即使在大型语言模型的兴起下也是如此。在本文中，我们介绍了Bounty Hunter，这是一种自动化的对手模拟方法，作为流行的对手模拟平台Caldera的开源插件设计和实现，它能够自主模拟具有多方面行为的对手，同时提供广泛的战术覆盖范围。为此，它实现了多样化的对抗行为，例如不同级别的可检测性和重复模拟中的不同攻击路径。通过自主破坏模拟企业网络，Bounty Hunter展示了其在不了解其目标的情况下实现给定目标的能力，包括破坏前、初始破坏和破坏后攻击策略。总体而言，Bounty Hunter促进了自主、全面和多方面的对手模拟，以帮助研究人员和从业者执行现实且省时的安全评估、培训练习和入侵检测研究。



## **8. An Efficient Gradient-Based Inference Attack for Federated Learning**

一种有效的联邦学习基于对象的推理攻击 cs.LG

This paper was supported by the TRUMPET project, funded by the European Union under Grant Agreement No. 101070038

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15143v1) [paper-pdf](https://arxiv.org/pdf/2512.15143v1)

**Authors**: Pablo Montaña-Fernández, Ines Ortega-Fernandez

**Abstract**: Federated Learning is a machine learning setting that reduces direct data exposure, improving the privacy guarantees of machine learning models. Yet, the exchange of model updates between the participants and the aggregator can still leak sensitive information. In this work, we present a new gradient-based membership inference attack for federated learning scenarios that exploits the temporal evolution of last-layer gradients across multiple federated rounds. Our method uses the shadow technique to learn round-wise gradient patterns of the training records, requiring no access to the private dataset, and is designed to consider both semi-honest and malicious adversaries (aggregators or data owners). Beyond membership inference, we also provide a natural extension of the proposed attack to discrete attribute inference by contrasting gradient responses under alternative attribute hypotheses. The proposed attacks are model-agnostic, and therefore applicable to any gradient-based model and can be applied to both classification and regression settings. We evaluate the attack on CIFAR-100 and Purchase100 datasets for membership inference and on Breast Cancer Wisconsin for attribute inference. Our findings reveal strong attack performance and comparable computational and memory overhead in membership inference when compared to another attack from the literature. The obtained results emphasize that multi-round federated learning can increase the vulnerability to inference attacks, that aggregators pose a more substantial threat than data owners, and that attack performance is strongly influenced by the nature of the training dataset, with richer, high-dimensional data leading to stronger leakage than simpler tabular data.

摘要: 联合学习是一种机器学习设置，可以减少直接数据暴露，改善机器学习模型的隐私保证。然而，参与者和聚合器之间的模型更新交换仍然可能泄露敏感信息。在这项工作中，我们针对联邦学习场景提出了一种新的基于梯度的成员资格推断攻击，该攻击利用了多个联邦回合中最后一层梯度的时间演变。我们的方法使用阴影技术来学习训练记录的全方位梯度模式，不需要访问私人数据集，并且旨在考虑半诚实和恶意对手（聚合器或数据所有者）。除了成员资格推断，我们还提供了一个自然的扩展，提出的攻击离散属性推理对比梯度响应下的替代属性假设。所提出的攻击是与模型无关的，因此适用于任何基于梯度的模型，并且可以应用于分类和回归设置。我们评估了对CIFAR-100和Purchase 100数据集的攻击，用于成员推断和对乳腺癌威斯康星州的属性推断。我们的研究结果揭示了强大的攻击性能和可比的计算和内存开销的成员推断相比，从文献中的另一种攻击。所获得的结果强调，多轮联邦学习可能会增加推理攻击的脆弱性，聚合器比数据所有者构成更大的威胁，攻击性能受到训练数据集性质的强烈影响，更丰富、多维的数据会导致比更简单的表格数据更强的泄漏。



## **9. Quantifying Return on Security Controls in LLM Systems**

量化LLM系统中安全控制的回报 cs.CR

13 pages, 9 figures, 3 tables

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15081v1) [paper-pdf](https://arxiv.org/pdf/2512.15081v1)

**Authors**: Richard Helder Moulton, Austin O'Brien, John D. Hastings

**Abstract**: Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).

摘要: 尽管大型语言模型（LLM）越来越多地用于安全关键工作流程，但从业者缺乏关于哪些保障措施值得部署的量化指导。本文介绍了一个面向决策的框架和可重复的方法论，它们共同量化剩余风险，将对抗性调查结果转化为财务风险估计和控制回报（RoC）指标，并实现基于LLM的系统的分层防御的货币比较。检索增强生成（RAG）服务使用DeepSeek-R1模型在包含合成个人可识别信息（PRI）的数据库上实例化，并在五个漏洞类别上受到Garak的自动攻击：PIP泄露、潜在上下文注入、提示注入、对抗攻击生成和分歧。对于每个（漏洞、控制）对，攻击成功概率是通过拉普拉斯继承规则估计的，并结合根据公共违规成本数据校准的损失三角分布，在10，000次运行的蒙特卡洛模拟中生成损失延迟曲线和预期损失。然后将三种广泛使用的缓解措施：基于属性的访问控制（ABAC）;使用Microsoft Presidio的命名实体识别（NER）编辑;和NeMo Guardrails与基线RAG配置进行比较。基线系统表现出非常高的攻击成功率（PRI、潜伏注射和即时注射>= 0.98），每个攻击场景的模拟预期损失总额为31.3万美元。ABAC将PRI和预算相关攻击的成功概率降至接近零，并将总预期损失降低约94%，实现了9.83的RoC。NER编辑同样消除了PRI泄漏，并获得了5.97的RoC，而NeMo Guardrails仅提供了边际效益（RoC为0.05）。



## **10. Cloud Security Leveraging AI: A Fusion-Based AISOC for Malware and Log Behaviour Detection**

利用人工智能的云安全：基于融合的AISOC，用于恶意软件和日志行为检测 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14935v1) [paper-pdf](https://arxiv.org/pdf/2512.14935v1)

**Authors**: Nnamdi Philip Okonkwo, Lubna Luxmi Dhirani

**Abstract**: Cloud Security Operations Center (SOC) enable cloud governance, risk and compliance by providing insights visibility and control. Cloud SOC triages high-volume, heterogeneous telemetry from elastic, short-lived resources while staying within tight budgets. In this research, we implement an AI-Augmented Security Operations Center (AISOC) on AWS that combines cloud-native instrumentation with ML-based detection. The architecture uses three Amazon EC2 instances: Attacker, Defender, and Monitoring. We simulate a reverse-shell intrusion with Metasploit, and Filebeat forwards Defender logs to an Elasticsearch and Kibana stack for analysis. We train two classifiers, a malware detector built on a public dataset and a log-anomaly detector trained on synthetically augmented logs that include adversarial variants. We calibrate and fuse the scores to produce multi-modal threat intelligence and triage activity into NORMAL, SUSPICIOUS, and HIGH\_CONFIDENCE\_ATTACK. On held-out tests the fusion achieves strong macro-F1 (up to 1.00) under controlled conditions, though performance will vary in noisier and more diverse environments. These results indicate that simple, calibrated fusion can enhance cloud SOC capabilities in constrained, cost-sensitive setups.

摘要: 云安全运营中心（SOC）通过提供洞察、可见性和控制来实现云治理、风险和合规性。云SOC从弹性、短暂的资源中对大容量、异类遥感进行分类，同时保持在紧张的预算范围内。在这项研究中，我们在AWS上实现了人工智能增强安全运营中心（AISOC），该中心将云原生工具与基于ML的检测相结合。该架构使用三个Amazon EC 2实例：Attacker、Defender和Monitoring。我们使用Metasploit模拟反向Shell入侵，Filebat将Defender日志转发到Elasticsearch和Kibana堆栈进行分析。我们训练两个分类器，一个是基于公共数据集构建的恶意软件检测器，另一个是基于包括对抗性变体的合成增强日志训练的日志异常检测器。我们校准和融合分数，以产生多模式威胁情报，并将活动分类为正常、可疑和高\_信心\_攻击。在进行测试中，融合在受控条件下实现了强大的宏F1（高达1.00），尽管性能在噪音更大和更多样化的环境中会有所不同。这些结果表明，简单的校准融合可以在受约束、成本敏感的设置中增强云SOC能力。



## **11. PerProb: Indirectly Evaluating Memorization in Large Language Models**

PerProb：间接评估大型语言模型中的精简化 cs.CR

Accepted at APSEC 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14600v1) [paper-pdf](https://arxiv.org/pdf/2512.14600v1)

**Authors**: Yihan Liao, Jacky Keung, Xiaoxue Ma, Jingyu Zhang, Yicheng Sun

**Abstract**: The rapid advancement of Large Language Models (LLMs) has been driven by extensive datasets that may contain sensitive information, raising serious privacy concerns. One notable threat is the Membership Inference Attack (MIA), where adversaries infer whether a specific sample was used in model training. However, the true impact of MIA on LLMs remains unclear due to inconsistent findings and the lack of standardized evaluation methods, further complicated by the undisclosed nature of many LLM training sets. To address these limitations, we propose PerProb, a unified, label-free framework for indirectly assessing LLM memorization vulnerabilities. PerProb evaluates changes in perplexity and average log probability between data generated by victim and adversary models, enabling an indirect estimation of training-induced memory. Compared with prior MIA methods that rely on member/non-member labels or internal access, PerProb is independent of model and task, and applicable in both black-box and white-box settings. Through a systematic classification of MIA into four attack patterns, we evaluate PerProb's effectiveness across five datasets, revealing varying memory behaviors and privacy risks among LLMs. Additionally, we assess mitigation strategies, including knowledge distillation, early stopping, and differential privacy, demonstrating their effectiveness in reducing data leakage. Our findings offer a practical and generalizable framework for evaluating and improving LLM privacy.

摘要: 大型语言模型（LLM）的快速发展是由可能包含敏感信息的大量数据集推动的，从而引发了严重的隐私问题。一个值得注意的威胁是会员推断攻击（MIA），对手推断特定样本是否用于模型训练。然而，由于调查结果不一致和缺乏标准化的评估方法，MIA对LLM的真正影响仍然不清楚，而且许多LLM培训集的未公开性质使其更加复杂。为了解决这些限制，我们提出了PerProb，这是一个统一的、无标签的框架，用于间接评估LLM记忆漏洞。PerProb评估受害者和对手模型生成的数据之间的困惑度和平均日志概率的变化，从而能够间接估计训练诱导的记忆。与之前依赖成员/非成员标签或内部访问的MIA方法相比，PerProb独立于模型和任务，适用于黑盒和白盒设置。通过将MIA系统地分类为四种攻击模式，我们评估了PerProb在五个数据集中的有效性，揭示了LLM之间不同的记忆行为和隐私风险。此外，我们还评估了缓解策略，包括知识提炼、提前停止和差异隐私，证明它们在减少数据泄露方面的有效性。我们的研究结果为评估和改善LLM隐私提供了一个实用和可推广的框架。



## **12. Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space**

通过隐形方式转移对LLM代理的推理方式中毒：RSV空间中的过程级攻击和收件箱监控 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14448v1) [paper-pdf](https://arxiv.org/pdf/2512.14448v1)

**Authors**: Xingfu Zhou, Pengfei Wang

**Abstract**: Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.

摘要: 依赖外部检索的大型语言模型（LLM）代理越来越多地部署在高风险环境中。虽然现有的对抗性攻击主要集中在内容伪造或指令注入上，但我们发现了一种新颖的、面向过程的攻击表面：代理的推理风格。我们提出了推理式中毒（RSP），这是一种操纵代理如何处理信息而不是处理内容的范式。我们引入了生成风格注入（GSI），这是一种攻击方法，将检索到的文档改写为病理性语气--特别是“分析瘫痪”或“认知仓促”--而无需改变基本事实或使用明确的触发器。为了量化这些转变，我们开发了推理风格载体（RSV），这是一种跟踪验证深度、自信和注意力焦点的指标。使用ReAct、ReReReflection和Tree of Thoughts（ToT）架构对HotpotQA和FEVER进行的实验表明，GSI会显着降低性能。它将推理步骤增加多达4.4倍，否则会导致过早错误，从而成功绕过最先进的内容过滤器。最后，我们提出了RSP-M，这是一种轻量级的运行时监视器，可实时计算RSV指标，并在值超过安全阈值时触发警报。我们的工作表明，推理风格是一个独特的、可利用的漏洞，需要静态内容分析之外的流程级防御。



## **13. Mimicking Human Visual Development for Learning Robust Image Representations**

模仿人类视觉发展以学习稳健的图像表示 cs.CV

Accepted to ICVGIP 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14360v1) [paper-pdf](https://arxiv.org/pdf/2512.14360v1)

**Authors**: Ankita Raj, Kaashika Prajaapat, Tapan Kumar Gandhi, Chetan Arora

**Abstract**: The human visual system is remarkably adept at adapting to changes in the input distribution; a capability modern convolutional neural networks (CNNs) still struggle to match. Drawing inspiration from the developmental trajectory of human vision, we propose a progressive blurring curriculum to improve the generalization and robustness of CNNs. Human infants are born with poor visual acuity, gradually refining their ability to perceive fine details. Mimicking this process, we begin training CNNs on highly blurred images during the initial epochs and progressively reduce the blur as training advances. This approach encourages the network to prioritize global structures over high-frequency artifacts, improving robustness against distribution shifts and noisy inputs. Challenging prior claims that blurring in the initial training epochs imposes a stimulus deficit and irreversibly harms model performance, we reveal that early-stage blurring enhances generalization with minimal impact on in-domain accuracy. Our experiments demonstrate that the proposed curriculum reduces mean corruption error (mCE) by up to 8.30% on CIFAR-10-C and 4.43% on ImageNet-100-C datasets, compared to standard training without blurring. Unlike static blur-based augmentation, which applies blurred images randomly throughout training, our method follows a structured progression, yielding consistent gains across various datasets. Furthermore, our approach complements other augmentation techniques, such as CutMix and MixUp, and enhances both natural and adversarial robustness against common attack methods. Code is available at https://github.com/rajankita/Visual_Acuity_Curriculum.

摘要: 人类视觉系统非常善于适应输入分布的变化;现代卷积神经网络（CNN）仍然难以匹敌的能力。我们从人类视觉的发展轨迹中汲取灵感，提出了一种渐进式模糊课程，以提高CNN的概括性和稳健性。人类婴儿生来视力就差，逐渐提高他们感知细微细节的能力。模仿这个过程，我们开始在初始时期对高度模糊的图像进行CNN训练，并随着训练的进展逐渐减少模糊。这种方法鼓励网络优先考虑全球结构而不是高频伪影，从而提高针对分布变化和有噪输入的鲁棒性。我们反驳了之前的说法，即初始训练时期的模糊会造成刺激赤字并不可逆转地损害模型性能，我们发现早期模糊会增强概括性，对域内准确性的影响最小。我们的实验表明，与没有模糊的标准训练相比，拟议的课程在CIFAR-10-C数据集上将平均腐败错误（mCE）减少了高达8.30%，在ImageNet-100-C数据集上将平均腐败错误（mCE）减少了4.43%。与基于模糊的静态增强（在整个训练过程中随机应用模糊图像）不同，我们的方法遵循结构化进程，在各种数据集中产生一致的收益。此外，我们的方法还补充了CutMix和MixUp等其他增强技术，并增强了针对常见攻击方法的自然和对抗鲁棒性。代码可在https://github.com/rajankita/Visual_Acuity_Curriculum上获取。



## **14. Optimizing the Adversarial Perturbation with a Momentum-based Adaptive Matrix**

利用基于动量的自适应矩阵优化对抗扰动 cs.LG

IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14188v1) [paper-pdf](https://arxiv.org/pdf/2512.14188v1)

**Authors**: Wei Tao, Sheng Long, Xin Liu, Wei Li, Qing Tao

**Abstract**: Generating adversarial examples (AEs) can be formulated as an optimization problem. Among various optimization-based attacks, the gradient-based PGD and the momentum-based MI-FGSM have garnered considerable interest. However, all these attacks use the sign function to scale their perturbations, which raises several theoretical concerns from the point of view of optimization. In this paper, we first reveal that PGD is actually a specific reformulation of the projected gradient method using only the current gradient to determine its step-size. Further, we show that when we utilize a conventional adaptive matrix with the accumulated gradients to scale the perturbation, PGD becomes AdaGrad. Motivated by this analysis, we present a novel momentum-based attack AdaMI, in which the perturbation is optimized with an interesting momentum-based adaptive matrix. AdaMI is proved to attain optimal convergence for convex problems, indicating that it addresses the non-convergence issue of MI-FGSM, thereby ensuring stability of the optimization process. The experiments demonstrate that the proposed momentum-based adaptive matrix can serve as a general and effective technique to boost adversarial transferability over the state-of-the-art methods across different networks while maintaining better stability and imperceptibility.

摘要: 生成对抗性示例（AE）可以被公式化为一个优化问题。在各种基于优化的攻击中，基于梯度的PGD和基于动量的MI-FGSM已经引起了相当大的兴趣。然而，所有这些攻击都使用符号函数来缩放其扰动，这从优化的角度提出了一些理论问题。在本文中，我们首先揭示，PGD实际上是一个特定的投影梯度方法的改造，只使用当前梯度来确定其步长。此外，我们表明，当我们利用具有累积梯度的传统自适应矩阵来缩放扰动时，PVD就会变成AdaGrad。受此分析的启发，我们提出了一种新型的基于动量的攻击AdaMI，其中使用一个有趣的基于动量的自适应矩阵对扰动进行了优化。AdaMI被证明能够实现凸问题的最优收敛，表明它解决了MI-FGSM的非收敛问题，从而确保了优化过程的稳定性。实验表明，所提出的基于动量的自适应矩阵可以作为一种通用且有效的技术，与最先进的方法相比，可以提高不同网络之间的对抗可转移性，同时保持更好的稳定性和不可感知性。



## **15. On Improving Deep Active Learning with Formal Verification**

利用形式验证改进深度主动学习 cs.LG

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14170v1) [paper-pdf](https://arxiv.org/pdf/2512.14170v1)

**Authors**: Jonathan Spiegelman, Guy Amir, Guy Katz

**Abstract**: Deep Active Learning (DAL) aims to reduce labeling costs in neural-network training by prioritizing the most informative unlabeled samples for annotation. Beyond selecting which samples to label, several DAL approaches further enhance data efficiency by augmenting the training set with synthetic inputs that do not require additional manual labeling. In this work, we investigate how augmenting the training data with adversarial inputs that violate robustness constraints can improve DAL performance. We show that adversarial examples generated via formal verification contribute substantially more than those produced by standard, gradient-based attacks. We apply this extension to multiple modern DAL techniques, as well as to a new technique that we propose, and show that it yields significant improvements in model generalization across standard benchmarks.

摘要: 深度主动学习（DAL）旨在通过优先考虑信息最丰富的未标记样本进行注释来降低神经网络训练中的标记成本。除了选择要标记的样本之外，几种DAL方法还通过使用不需要额外手动标记的合成输入来增强训练集，进一步提高了数据效率。在这项工作中，我们研究了如何用违反稳健性约束的对抗输入来增强训练数据可以提高DAL性能。我们表明，通过形式验证生成的对抗性示例的贡献远高于通过标准的基于梯度的攻击生成的对抗性示例。我们将此扩展应用于多种现代DAL技术以及我们提出的一种新技术，并表明它在标准基准测试中的模型概括方面产生了显着改进。



## **16. MURIM: Multidimensional Reputation-based Incentive Mechanism for Federated Learning**

MURIM：基于声誉的多维联邦学习激励机制 cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13955v1) [paper-pdf](https://arxiv.org/pdf/2512.13955v1)

**Authors**: Sindhuja Madabushi, Dawood Wasif, Jin-Hee Cho

**Abstract**: Federated Learning (FL) has emerged as a leading privacy-preserving machine learning paradigm, enabling participants to share model updates instead of raw data. However, FL continues to face key challenges, including weak client incentives, privacy risks, and resource constraints. Assessing client reliability is essential for fair incentive allocation and ensuring that each client's data contributes meaningfully to the global model. To this end, we propose MURIM, a MUlti-dimensional Reputation-based Incentive Mechanism that jointly considers client reliability, privacy, resource capacity, and fairness while preventing malicious or unreliable clients from earning undeserved rewards. MURIM allocates incentives based on client contribution, latency, and reputation, supported by a reliability verification module. Extensive experiments on MNIST, FMNIST, and ADULT Income datasets demonstrate that MURIM achieves up to 18% improvement in fairness metrics, reduces privacy attack success rates by 5-9%, and improves robustness against poisoning and noisy-gradient attacks by up to 85% compared to state-of-the-art baselines. Overall, MURIM effectively mitigates adversarial threats, promotes fair and truthful participation, and preserves stable model convergence across heterogeneous and dynamic federated settings.

摘要: 联合学习（FL）已成为一种领先的隐私保护机器学习范式，使参与者能够共享模型更新而不是原始数据。然而，FL继续面临关键挑战，包括客户激励薄弱、隐私风险和资源限制。评估客户可靠性对于公平的激励分配和确保每个客户的数据对全球模型做出有意义的贡献至关重要。为此，我们提出了MURIM，这是一种多维基于声誉的激励机制，它联合考虑客户可靠性、隐私、资源容量和公平性，同时防止恶意或不可靠的客户获得不应有的回报。MURIM根据客户贡献、延迟和声誉分配激励，并由可靠性验证模块支持。对MNIST、FMNIST和ADSYS Income数据集的广泛实验表明，与最先进的基线相比，MURIM在公平性指标方面实现了高达18%的提高，将隐私攻击成功率降低5- 9%，并将针对中毒和噪音梯度攻击的鲁棒性提高了高达85%。总体而言，MURIM有效地缓解了对抗威胁，促进了公平和真实的参与，并在异类和动态联邦环境中保持稳定的模型融合。



## **17. Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures (XAMT)**

异类多代理体系结构（XAMT）中隐蔽内存篡改的双层优化 cs.CR

10 pages, 5 figures, 4 tables. Conference-style paper (IEEEtran). Proposes unified bilevel optimization framework for covert memory poisoning attacks in heterogeneous multi-agent systems (MARL + RAG)

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.15790v1) [paper-pdf](https://arxiv.org/pdf/2512.15790v1)

**Authors**: Akhil Sharma, Shaikh Yaser Arafat, Jai Kumar Sharma, Ken Huang

**Abstract**: The increasing operational reliance on complex Multi-Agent Systems (MAS) across safety-critical domains necessitates rigorous adversarial robustness assessment. Modern MAS are inherently heterogeneous, integrating conventional Multi-Agent Reinforcement Learning (MARL) with emerging Large Language Model (LLM) agent architectures utilizing Retrieval-Augmented Generation (RAG). A critical shared vulnerability is reliance on centralized memory components: the shared Experience Replay (ER) buffer in MARL and the external Knowledge Base (K) in RAG agents. This paper proposes XAMT (Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures), a novel framework that formalizes attack generation as a bilevel optimization problem. The Upper Level minimizes perturbation magnitude (delta) to enforce covertness while maximizing system behavior divergence toward an adversary-defined target (Lower Level). We provide rigorous mathematical instantiations for CTDE MARL algorithms and RAG-based LLM agents, demonstrating that bilevel optimization uniquely crafts stealthy, minimal-perturbation poisons evading detection heuristics. Comprehensive experimental protocols utilize SMAC and SafeRAG benchmarks to quantify effectiveness at sub-percent poison rates (less than or equal to 1 percent in MARL, less than or equal to 0.1 percent in RAG). XAMT defines a new unified class of training-time threats essential for developing intrinsically secure MAS, with implications for trust, formal verification, and defensive strategies prioritizing intrinsic safety over perimeter-based detection.

摘要: 安全关键领域对复杂多智能体系统（MAS）的运营依赖日益增加，需要进行严格的对抗稳健性评估。现代MAS本质上是异类的，将传统的多智能体强化学习（MARL）与利用检索增强生成（RAG）的新兴大型语言模型（LLM）智能体架构集成在一起。一个关键的共享漏洞是对集中式内存组件的依赖：MARL中的共享体验重播（ER）缓冲区和RAG代理中的外部知识库（K）。本文提出了XAPT（在异类多代理体系结构中针对隐蔽内存篡改的双层优化），这是一个新颖的框架，将攻击生成形式化为双层优化问题。上级最小化扰动幅度（增量）以加强隐蔽性，同时最大化系统行为向敌对定义的目标（下级）的分歧。我们为CTDE MARL算法和基于RAG的LLM代理提供了严格的数学实例，证明两层优化独特地处理了逃避检测启发的隐形、最小扰动毒药。全面的实验方案利用SMAC和SafeRAG基准来量化次百分中毒率（MARL中小于或等于1%，RAG中小于或等于0.1%）的有效性。XAMT定义了一种新的统一训练时威胁，这对于开发本质安全的MAS至关重要，对信任、形式验证和防御策略产生了影响，这些策略优先于本质安全而不是基于边界的检测。



## **18. REVERB-FL: Server-Side Adversarial and Reserve-Enhanced Federated Learning for Robust Audio Classification**

REVERB-FL：用于稳健音频分类的服务器端对抗和保留增强联邦学习 eess.AS

13 pages, 4 figures

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13647v1) [paper-pdf](https://arxiv.org/pdf/2512.13647v1)

**Authors**: Sathwika Peechara, Rajeev Sahay

**Abstract**: Federated learning (FL) enables a privacy-preserving training paradigm for audio classification but is highly sensitive to client heterogeneity and poisoning attacks, where adversarially compromised clients can bias the global model and hinder the performance of audio classifiers. To mitigate the effects of model poisoning for audio signal classification, we present REVERB-FL, a lightweight, server-side defense that couples a small reserve set (approximately 5%) with pre- and post-aggregation retraining and adversarial training. After each local training round, the server refines the global model on the reserve set with either clean or additional adversarially perturbed data, thereby counteracting non-IID drift and mitigating potential model poisoning without adding substantial client-side cost or altering the aggregation process. We theoretically demonstrate the feasibility of our framework, showing faster convergence and a reduced steady-state error relative to baseline federated averaging. We validate our framework on two open-source audio classification datasets with varying IID and Dirichlet non-IID partitions and demonstrate that REVERB-FL mitigates global model poisoning under multiple designs of local data poisoning.

摘要: 联合学习（FL）为音频分类提供了一种保护隐私的训练范式，但对客户端的多样性和中毒攻击高度敏感，其中敌对妥协的客户端可能会对全局模型产生偏差并阻碍音频分类器的性能。为了减轻模型中毒对音频信号分类的影响，我们提出了REVERB-FL，这是一种轻量级的服务器端防御，它将小的储备集（约5%）与聚集前和聚集后的再训练和对抗训练结合起来。每次本地训练轮结束后，服务器都会用干净的或额外的敌对扰动数据来细化储备集中的全局模型，从而抵消非IID漂移并减轻潜在的模型中毒，而不会增加大量客户端成本或改变聚合过程。我们从理论上证明了我们框架的可行性，相对于基线联邦平均，收敛速度更快，稳态误差更小。我们在具有不同IID和Dirichlet非IID分区的两个开源音频分类数据集上验证了我们的框架，并证明REVERB-FL可以缓解多种本地数据中毒设计下的全局模型中毒。



## **19. Async Control: Stress-testing Asynchronous Control Measures for LLM Agents**

同步控制：LLM代理的压力测试同步控制措施 cs.LG

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13526v1) [paper-pdf](https://arxiv.org/pdf/2512.13526v1)

**Authors**: Asa Cooper Stickland, Jan Michelfeit, Arathi Mani, Charlie Griffin, Ollie Matthews, Tomek Korbak, Rogan Inglis, Oliver Makins, Alan Cooney

**Abstract**: LLM-based software engineering agents are increasingly used in real-world development tasks, often with access to sensitive data or security-critical codebases. Such agents could intentionally sabotage these codebases if they were misaligned. We investigate asynchronous monitoring, in which a monitoring system reviews agent actions after the fact. Unlike synchronous monitoring, this approach does not impose runtime latency, while still attempting to disrupt attacks before irreversible harm occurs. We treat monitor development as an adversarial game between a blue team (who design monitors) and a red team (who create sabotaging agents). We attempt to set the game rules such that they upper bound the sabotage potential of an agent based on Claude 4.1 Opus. To ground this game in a realistic, high-stakes deployment scenario, we develop a suite of 5 diverse software engineering environments that simulate tasks that an agent might perform within an AI developer's internal infrastructure. Over the course of the game, we develop an ensemble monitor that achieves a 6% false negative rate at 1% false positive rate on a held out test environment. Then, we estimate risk of sabotage at deployment time by extrapolating from our monitor's false negative rate. We describe one simple model for this extrapolation, present a sensitivity analysis, and describe situations in which the model would be invalid. Code is available at: https://github.com/UKGovernmentBEIS/async-control.

摘要: 基于LLM的软件工程代理越来越多地用于现实世界的开发任务，通常可以访问敏感数据或安全关键代码库。如果这些代码库未对齐，此类代理可能会故意破坏它们。我们研究了同步监控，其中监控系统在事后审查代理动作。与同步监控不同，这种方法不会施加运行时延迟，同时仍试图在不可逆转的伤害发生之前中断攻击。我们将显示器开发视为蓝色团队（设计显示器）和红色团队（创建破坏性代理）之间的对抗游戏。我们试图设定游戏规则，使其上限基于Claude 4.1 Opus的代理的破坏潜力。为了将这款游戏置于现实、高风险的部署场景中，我们开发了一套由5个不同的软件工程环境，这些环境模拟代理可能在人工智能开发人员的内部基础设施中执行的任务。在游戏过程中，我们开发了一个整体监视器，可以在固定的测试环境中实现6%的假阴性率和1%的假阳性率。然后，我们通过从监视器的假阴性率推断来估计部署时破坏的风险。我们描述了一个用于此外推的简单模型，提出了敏感性分析，并描述了模型无效的情况。代码可访问：https://github.com/UKGovernmentBEIS/async-control。



## **20. An $H_2$-norm approach to performance analysis of networked control systems under multiplicative routing transformations**

乘性路由变换下网络控制系统性能分析的$H_2$-模方法 eess.SY

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13504v1) [paper-pdf](https://arxiv.org/pdf/2512.13504v1)

**Authors**: Ruslan Seifullaev, André M. H. Teixeira

**Abstract**: This paper investigates the performance of networked control systems subject to multiplicative routing transformations that alter measurement pathways without directly injecting signals. Such transformations, arising from faults or adversarial actions, modify the feedback structure and can degrade performance while remaining stealthy. An $H_2$-norm framework is proposed to quantify the impact of these transformations by evaluating the ratio between the steady-state energies of performance and residual outputs. Equivalent linear matrix inequality (LMI) formulations are derived for computational assessment, and analytical upper bounds are established to estimate the worst-case degradation. The results provide structural insight into how routing manipulations influence closed-loop behavior and reveal conditions for stealthy multiplicative attacks.

摘要: 本文研究了经历乘性路由变换的网络控制系统的性能，该变换可以在不直接注入信号的情况下改变测量路径。这种由故障或对抗行为引起的转换会修改反馈结构，并可能会降低性能，同时保持隐形。提出了一个$H_2$-norm框架，通过评估性能的稳态能量与剩余产出之间的比率来量化这些转换的影响。推导出用于计算评估的等效线性矩阵不等式（LGA）公式，并建立分析上界以估计最坏情况的退化。结果提供了结构性的洞察路由操作如何影响闭环行为，并揭示了隐形乘法攻击的条件。



## **21. Behavior-Aware and Generalizable Defense Against Black-Box Adversarial Attacks for ML-Based IDS**

针对基于ML的IDS的黑匣子对抗攻击的行为感知和可推广防御 cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13501v1) [paper-pdf](https://arxiv.org/pdf/2512.13501v1)

**Authors**: Sabrine Ennaji, Elhadj Benkhelifa, Luigi Vincenzo Mancini

**Abstract**: Machine learning based intrusion detection systems are increasingly targeted by black box adversarial attacks, where attackers craft evasive inputs using indirect feedback such as binary outputs or behavioral signals like response time and resource usage. While several defenses have been proposed, including input transformation, adversarial training, and surrogate detection, they often fall short in practice. Most are tailored to specific attack types, require internal model access, or rely on static mechanisms that fail to generalize across evolving attack strategies. Furthermore, defenses such as input transformation can degrade intrusion detection system performance, making them unsuitable for real time deployment.   To address these limitations, we propose Adaptive Feature Poisoning, a lightweight and proactive defense mechanism designed specifically for realistic black box scenarios. Adaptive Feature Poisoning assumes that probing can occur silently and continuously, and introduces dynamic and context aware perturbations to selected traffic features, corrupting the attacker feedback loop without impacting detection capabilities. The method leverages traffic profiling, change point detection, and adaptive scaling to selectively perturb features that an attacker is likely exploiting, based on observed deviations.   We evaluate Adaptive Feature Poisoning against multiple realistic adversarial attack strategies, including silent probing, transferability based attacks, and decision boundary based attacks. The results demonstrate its ability to confuse attackers, degrade attack effectiveness, and preserve detection performance. By offering a generalizable, attack agnostic, and undetectable defense, Adaptive Feature Poisoning represents a significant step toward practical and robust adversarial resilience in machine learning based intrusion detection systems.

摘要: 基于机器学习的入侵检测系统越来越成为黑匣子对抗攻击的目标，攻击者使用间接反馈（例如二进制输出或响应时间和资源使用等行为信号）来制造规避输入。虽然已经提出了多种防御措施，包括输入转换、对抗训练和代理检测，但它们在实践中往往达不到要求。大多数都是针对特定攻击类型定制的，需要内部模型访问，或者依赖于无法在不断发展的攻击策略中进行概括的静态机制。此外，输入转换等防御措施可能会降低入侵检测系统的性能，使其不适合实时部署。   为了解决这些限制，我们提出了自适应特征中毒，这是一种专门为现实黑匣子场景设计的轻量级主动防御机制。自适应特征中毒假设探测可以无声且连续地发生，并向选定的流量特征引入动态和上下文感知的扰动，从而破坏攻击者反馈循环，而不影响检测能力。该方法利用流量分析、变点检测和自适应缩放来根据观察到的偏差选择性地扰乱攻击者可能利用的特征。   我们针对多种现实的对抗攻击策略来评估自适应特征中毒，包括无声探测、基于可转移性的攻击和基于决策边界的攻击。结果表明，它能够混淆攻击者，降低攻击效果，并保持检测性能。通过提供可推广的，攻击不可知的和不可检测的防御，自适应特征中毒代表了在基于机器学习的入侵检测系统中实现实用和强大的对抗弹性的重要一步。



## **22. On the Effectiveness of Membership Inference in Targeted Data Extraction from Large Language Models**

隶属推理在大型语言模型有针对性数据提取中的有效性 cs.LG

Accepted to IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13352v1) [paper-pdf](https://arxiv.org/pdf/2512.13352v1)

**Authors**: Ali Al Sahili, Ali Chehab, Razane Tajeddine

**Abstract**: Large Language Models (LLMs) are prone to memorizing training data, which poses serious privacy risks. Two of the most prominent concerns are training data extraction and Membership Inference Attacks (MIAs). Prior research has shown that these threats are interconnected: adversaries can extract training data from an LLM by querying the model to generate a large volume of text and subsequently applying MIAs to verify whether a particular data point was included in the training set. In this study, we integrate multiple MIA techniques into the data extraction pipeline to systematically benchmark their effectiveness. We then compare their performance in this integrated setting against results from conventional MIA benchmarks, allowing us to evaluate their practical utility in real-world extraction scenarios.

摘要: 大型语言模型（LLM）容易记住训练数据，这会带来严重的隐私风险。两个最突出的问题是训练数据提取和成员推断攻击（MIA）。之前的研究表明，这些威胁是相互关联的：对手可以通过查询模型以生成大量文本，然后应用MIA来验证特定数据点是否包含在训练集中，从而从LLM中提取训练数据。在这项研究中，我们将多种MIA技术集成到数据提取管道中，以系统地衡量其有效性。然后，我们将它们在此集成环境中的性能与传统MIA基准的结果进行比较，使我们能够评估它们在现实世界提取场景中的实际实用性。



## **23. Evaluating Adversarial Attacks on Federated Learning for Temperature Forecasting**

评估温度预测联邦学习的对抗攻击 cs.LG

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.13207v2) [paper-pdf](https://arxiv.org/pdf/2512.13207v2)

**Authors**: Karina Chichifoi, Fabio Merizzi, Michele Colajanni

**Abstract**: Deep learning and federated learning (FL) are becoming powerful partners for next-generation weather forecasting. Deep learning enables high-resolution spatiotemporal forecasts that can surpass traditional numerical models, while FL allows institutions in different locations to collaboratively train models without sharing raw data, addressing efficiency and security concerns. While FL has shown promise across heterogeneous regions, its distributed nature introduces new vulnerabilities. In particular, data poisoning attacks, in which compromised clients inject manipulated training data, can degrade performance or introduce systematic biases. These threats are amplified by spatial dependencies in meteorological data, allowing localized perturbations to influence broader regions through global model aggregation. In this study, we investigate how adversarial clients distort federated surface temperature forecasts trained on the Copernicus European Regional ReAnalysis (CERRA) dataset. We simulate geographically distributed clients and evaluate patch-based and global biasing attacks on regional temperature forecasts. Our results show that even a small fraction of poisoned clients can mislead predictions across large, spatially connected areas. A global temperature bias attack from a single compromised client shifts predictions by up to -1.7 K, while coordinated patch attacks more than triple the mean squared error and produce persistent regional anomalies exceeding +3.5 K. Finally, we assess trimmed mean aggregation as a defense mechanism, showing that it successfully defends against global bias attacks (2-13% degradation) but fails against patch attacks (281-603% amplification), exposing limitations of outlier-based defenses for spatially correlated data.

摘要: 深度学习和联合学习（FL）正在成为下一代天气预报的强大合作伙伴。深度学习可以实现超越传统数值模型的高分辨率时空预测，而FL则允许不同地点的机构在无需共享原始数据的情况下协作训练模型，从而解决效率和安全问题。虽然FL在不同地区表现出了希望，但其分布式性质带来了新的漏洞。特别是，数据中毒攻击（即受影响的客户端注入操纵的训练数据）可能会降低性能或引入系统性偏差。气象数据的空间依赖性放大了这些威胁，使局部扰动能够通过全球模型聚合影响更广泛的区域。在这项研究中，我们调查了对抗性客户端如何扭曲在哥白尼欧洲区域再分析（CERRA）数据集上训练的联合表面温度预测。我们模拟地理上分布的客户端和评估补丁为基础的和全球偏见的攻击区域温度预报。我们的研究结果表明，即使是一小部分中毒的客户端也会误导大型空间连接区域的预测。来自单个受损客户端的全球温度偏差攻击将预测值改变高达-1.7 K，而协调补丁攻击则是均方误差的三倍多，并产生超过+3.5 K的持续区域异常。最后，我们评估了修剪平均聚集作为防御机制，表明它成功防御全局偏差攻击（2-13%降级），但未能防御补丁攻击（281-603%放大），暴露了基于离群值的防御空间相关数据的局限性。



## **24. Less Is More: Sparse and Cooperative Perturbation for Point Cloud Attacks**

少即是多：点云攻击的稀疏和合作扰动 cs.CR

Accepted by AAAI'2026 (Oral)

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13119v1) [paper-pdf](https://arxiv.org/pdf/2512.13119v1)

**Authors**: Keke Tang, Tianyu Hao, Xiaofei Wang, Weilong Peng, Denghui Zhang, Peican Zhu, Zhihong Tian

**Abstract**: Most adversarial attacks on point clouds perturb a large number of points, causing widespread geometric changes and limiting applicability in real-world scenarios. While recent works explore sparse attacks by modifying only a few points, such approaches often struggle to maintain effectiveness due to the limited influence of individual perturbations. In this paper, we propose SCP, a sparse and cooperative perturbation framework that selects and leverages a compact subset of points whose joint perturbations produce amplified adversarial effects. Specifically, SCP identifies the subset where the misclassification loss is locally convex with respect to their joint perturbations, determined by checking the positivedefiniteness of the corresponding Hessian block. The selected subset is then optimized to generate high-impact adversarial examples with minimal modifications. Extensive experiments show that SCP achieves 100% attack success rates, surpassing state-of-the-art sparse attacks, and delivers superior imperceptibility to dense attacks with far fewer modifications.

摘要: 对点云的大多数对抗攻击都会扰乱大量点，导致广泛的几何变化并限制其在现实世界场景中的适用性。虽然最近的作品仅通过修改几个点来探索稀疏攻击，但由于个体扰动的影响有限，此类方法通常难以维持有效性。在本文中，我们提出了SCP，这是一种稀疏和合作的扰动框架，它选择和利用点的紧凑子集，这些点的联合扰动会产生放大的对抗效应。具体来说，SCP识别错误分类损失相对于其联合扰动为局部凸的子集，通过检查相应Hessian块的正定性来确定。然后对所选子集进行优化，以生成具有最小修改的高影响力对抗示例。大量实验表明，SCP的攻击成功率达到了100%，超越了最先进的稀疏攻击，并以更少的修改为密集攻击提供了卓越的不可感知性。



## **25. Calibrating Uncertainty for Zero-Shot Adversarial CLIP**

校准零镜头对抗CLIP的不确定性 cs.CV

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12997v1) [paper-pdf](https://arxiv.org/pdf/2512.12997v1)

**Authors**: Wenjing lu, Zerui Tao, Dongping Zhang, Yuning Qiu, Yang Yang, Qibin Zhao

**Abstract**: CLIP delivers strong zero-shot classification but remains highly vulnerable to adversarial attacks. Previous work of adversarial fine-tuning largely focuses on matching the predicted logits between clean and adversarial examples, which overlooks uncertainty calibration and may degrade the zero-shot generalization. A common expectation in reliable uncertainty estimation is that predictive uncertainty should increase as inputs become more difficult or shift away from the training distribution. However, we frequently observe the opposite in the adversarial setting: perturbations not only degrade accuracy but also suppress uncertainty, leading to severe miscalibration and unreliable over-confidence. This overlooked phenomenon highlights a critical reliability gap beyond robustness. To bridge this gap, we propose a novel adversarial fine-tuning objective for CLIP considering both prediction accuracy and uncertainty alignments. By reparameterizing the output of CLIP as the concentration parameter of a Dirichlet distribution, we propose a unified representation that captures relative semantic structure and the magnitude of predictive confidence. Our objective aligns these distributions holistically under perturbations, moving beyond single-logit anchoring and restoring calibrated uncertainty. Experiments on multiple zero-shot classification benchmarks demonstrate that our approach effectively restores calibrated uncertainty and achieves competitive adversarial robustness while maintaining clean accuracy.

摘要: CLIP提供了强大的零射击分类，但仍然极易受到对抗攻击。之前的对抗性微调工作主要集中在匹配干净和对抗性示例之间的预测逻辑，这忽视了不确定性校准，并可能降低零镜头概括性。可靠不确定性估计的一个常见期望是，随着输入变得更加困难或偏离训练分布，预测不确定性应该增加。然而，我们在对抗环境中经常观察到相反的情况：扰动不仅会降低准确性，还会抑制不确定性，导致严重的校准错误和不可靠的过度自信。这种被忽视的现象凸显了鲁棒性之外的关键可靠性差距。为了弥合这一差距，我们为CLIP提出了一种新颖的对抗性微调目标，同时考虑预测准确性和不确定性对齐。通过将CLIP的输出重新参数化为Dirichlet分布的浓度参数，我们提出了一个统一的表示，可以捕捉相对语义结构和预测置信度的大小。我们的目标在扰动下整体对齐这些分布，超越单logit锚定并恢复校准的不确定性。多个零镜头分类基准的实验表明，我们的方法有效地恢复了校准的不确定性，并在保持清晰准确性的同时实现了竞争对手鲁棒性。



## **26. The Eminence in Shadow: Exploiting Feature Boundary Ambiguity for Robust Backdoor Attacks**

阴影中的杰出：利用特征边界模糊性进行稳健的后门攻击 cs.LG

Accepted by KDD2026 Cycle 1 Research Track

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.10402v2) [paper-pdf](https://arxiv.org/pdf/2512.10402v2)

**Authors**: Zhou Feng, Jiahao Chen, Chunyi Zhou, Yuwen Pu, Tianyu Du, Jinbao Li, Jianhai Chen, Shouling Ji

**Abstract**: Deep neural networks (DNNs) underpin critical applications yet remain vulnerable to backdoor attacks, typically reliant on heuristic brute-force methods. Despite significant empirical advancements in backdoor research, the lack of rigorous theoretical analysis limits understanding of underlying mechanisms, constraining attack predictability and adaptability. Therefore, we provide a theoretical analysis targeting backdoor attacks, focusing on how sparse decision boundaries enable disproportionate model manipulation. Based on this finding, we derive a closed-form, ambiguous boundary region, wherein negligible relabeled samples induce substantial misclassification. Influence function analysis further quantifies significant parameter shifts caused by these margin samples, with minimal impact on clean accuracy, formally grounding why such low poison rates suffice for efficacious attacks. Leveraging these insights, we propose Eminence, an explainable and robust black-box backdoor framework with provable theoretical guarantees and inherent stealth properties. Eminence optimizes a universal, visually subtle trigger that strategically exploits vulnerable decision boundaries and effectively achieves robust misclassification with exceptionally low poison rates (< 0.1%, compared to SOTA methods typically requiring > 1%). Comprehensive experiments validate our theoretical discussions and demonstrate the effectiveness of Eminence, confirming an exponential relationship between margin poisoning and adversarial boundary manipulation. Eminence maintains > 90% attack success rate, exhibits negligible clean-accuracy loss, and demonstrates high transferability across diverse models, datasets and scenarios.

摘要: 深度神经网络（DNN）支撑关键应用程序，但仍然容易受到后门攻击，通常依赖于启发式暴力方法。尽管后门研究取得了重大的经验进展，但缺乏严格的理论分析限制了对潜在机制的理解，限制了攻击的可预测性和适应性。因此，我们提供了针对后门攻击的理论分析，重点关注稀疏决策边界如何导致不成比例的模型操纵。基于这一发现，我们得到了一个封闭形式的、模糊的边界区域，其中可忽略的重新标记样本会导致严重的误分类。影响函数分析进一步量化了这些边缘样本引起的显着参数变化，对清洁准确性的影响最小，正式证明了为什么如此低的中毒率足以进行有效的攻击。利用这些见解，我们提出了Eminence，这是一个可解释且强大的黑匣子后门框架，具有可证明的理论保证和固有的隐形属性。Eminence优化了一种通用的、视觉上微妙的触发器，该触发器战略性地利用脆弱的决策边界，并以极低的中毒率（<0.1%，而SOTA方法通常需要> 1%）有效地实现稳健的错误分类。全面的实验验证了我们的理论讨论，并证明了Eminence的有效性，证实了边缘中毒和对抗性边界操纵之间的指数关系。Eminence保持> 90%的攻击成功率，表现出可忽略不计的干净准确性损失，并表现出跨不同模型、数据集和场景的高可移植性。



## **27. Developing Distance-Aware, and Evident Uncertainty Quantification in Dynamic Physics-Constrained Neural Networks for Robust Bearing Degradation Estimation**

在动态物理约束神经网络中开发距离感知和明显不确定性量化，用于鲁棒轴承退化估计 cs.LG

Under review at Structural health Monitoring - SAGE

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.08499v2) [paper-pdf](https://arxiv.org/pdf/2512.08499v2)

**Authors**: Waleed Razzaq, Yun-Bo Zhao

**Abstract**: Accurate and uncertainty-aware degradation estimation is essential for predictive maintenance in safety-critical systems like rotating machinery with rolling-element bearings. Many existing uncertainty methods lack confidence calibration, are costly to run, are not distance-aware, and fail to generalize under out-of-distribution data. We introduce two distance-aware uncertainty methods for deterministic physics-guided neural networks: PG-SNGP, based on Spectral Normalization Gaussian Process, and PG-SNER, based on Deep Evidential Regression. We apply spectral normalization to the hidden layers so the network preserves distances from input to latent space. PG-SNGP replaces the final dense layer with a Gaussian Process layer for distance-sensitive uncertainty, while PG-SNER outputs Normal Inverse Gamma parameters to model uncertainty in a coherent probabilistic form. We assess performance using standard accuracy metrics and a new distance-aware metric based on the Pearson Correlation Coefficient, which measures how well predicted uncertainty tracks the distance between test and training samples. We also design a dynamic weighting scheme in the loss to balance data fidelity and physical consistency. We test our methods on rolling-element bearing degradation using the PRONOSTIA, XJTU-SY and HUST datasets and compare them with Monte Carlo and Deep Ensemble PGNNs. Results show that PG-SNGP and PG-SNER improve prediction accuracy, generalize reliably under OOD conditions, and remain robust to adversarial attacks and noise.

摘要: 准确且具有不确定性的退化估计对于具有滚动元件轴承的旋转机械等安全关键系统的预测性维护至关重要。许多现有的不确定性方法缺乏置信度校准、运行成本高、不具有距离意识，并且无法在非分布数据下进行概括。我们为确定性物理引导神经网络引入了两种距离感知不确定性方法：基于谱正规化高斯过程的PG-SNGP和基于深度证据回归的PG-SNER。我们对隐藏层应用光谱正规化，以便网络保留从输入到潜在空间的距离。PG-SNGP用高斯过程层取代最终的密集层，以实现距离敏感的不确定性，而PG-SNER输出正态逆伽玛参数，以连贯的概率形式对不确定性进行建模。我们使用标准准确性指标和基于皮尔逊相关系数的新距离感知指标来评估性能，皮尔逊相关系数衡量预测的不确定性跟踪测试和训练样本之间距离的程度。我们还设计了一个动态加权方案，以平衡数据保真度和物理一致性的损失。我们使用PRONOSTIA、XJTU-SY和HUST数据集测试我们关于滚动元件轴承退化的方法，并将其与Monte Carlo和Deep Ensemble PGNN进行比较。结果表明，PG-SNGP和PG-SNER提高了预测精度，在OOD条件下可靠地推广，并对对抗性攻击和噪声保持鲁棒性。



## **28. SEA: Spectral Edge Attack**

SEA：光谱边缘攻击 cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.08964v2) [paper-pdf](https://arxiv.org/pdf/2512.08964v2)

**Authors**: Yongyu Wang

**Abstract**: Graph based machine learning algorithms occupy an important position in today AI landscape. The ability of graph topology to represent complex data structures is both the key strength of graph algorithms and a source of their vulnerability. In other words, attacking or perturbing a graph can severely degrade the performance of graph-based methods. For the attack methods, the greatest challenge is achieving strong attack effectiveness while remaining undetected. To address this problem, this paper proposes a new attack model that employs spectral adversarial robustness evaluation to quantitatively analyze the vulnerability of each edge in a graph under attack. By precisely targeting the weakest links, the proposed approach achieves the maximum attack impact with minimal perturbation. Experimental results demonstrate the effectiveness of the proposed method.

摘要: 基于图的机器学习算法在当今人工智能领域占据重要地位。图布局表示复杂数据结构的能力既是图算法的关键优势，也是其脆弱性的根源。换句话说，攻击或干扰图可能会严重降低基于图的方法的性能。对于攻击方法来说，最大的挑战是在不被发现的情况下实现强大的攻击有效性。为了解决这个问题，本文提出了一种新的攻击模型，该模型采用谱对抗鲁棒性评估来定量分析受攻击图中每条边的脆弱性。通过精确地针对最弱的环节，提出的方法以最小的干扰实现了最大的攻击影响。实验结果证明了该方法的有效性。



## **29. SA$^{2}$GFM: Enhancing Robust Graph Foundation Models with Structure-Aware Semantic Augmentation**

SA$^{2}$GFM：通过结构感知语义增强稳健图基础模型 cs.LG

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.07857v2) [paper-pdf](https://arxiv.org/pdf/2512.07857v2)

**Authors**: Junhua Shi, Qingyun Sun, Haonan Yuan, Xingcheng Fu

**Abstract**: We present Graph Foundation Models (GFMs) which have made significant progress in various tasks, but their robustness against domain noise, structural perturbations, and adversarial attacks remains underexplored. A key limitation is the insufficient modeling of hierarchical structural semantics, which are crucial for generalization. In this paper, we propose SA$^{2}$GFM, a robust GFM framework that improves domain-adaptive representations through Structure-Aware Semantic Augmentation. First, we encode hierarchical structural priors by transforming entropy-based encoding trees into structure-aware textual prompts for feature augmentation. The enhanced inputs are processed by a self-supervised Information Bottleneck mechanism that distills robust, transferable representations via structure-guided compression. To address negative transfer in cross-domain adaptation, we introduce an expert adaptive routing mechanism, combining a mixture-of-experts architecture with a null expert design. For efficient downstream adaptation, we propose a fine-tuning module that optimizes hierarchical structures through joint intra- and inter-community structure learning. Extensive experiments demonstrate that SA$^{2}$GFM outperforms 9 state-of-the-art baselines in terms of effectiveness and robustness against random noise and adversarial perturbations for node and graph classification.

摘要: 我们提出的图形基础模型（GFM）在各种任务中取得了显着进展，但其对域噪音、结构扰动和对抗攻击的鲁棒性仍然没有得到充分的研究。一个关键限制是分层结构语义的建模不足，而分层结构语义对于概括至关重要。在本文中，我们提出SA$^{2}$GFM，这是一个强大的GFM框架，通过结构感知语义增强来改进域自适应表示。首先，我们通过将基于熵的编码树转换为结构感知的文本提示来编码分层结构先验以进行特征增强。增强的输入由自我监督的信息瓶颈机制处理，该机制通过结构引导压缩提取稳健的、可转移的表示。为了解决跨域适应中的负转移问题，我们引入了一种专家自适应路由机制，将混合专家架构与空专家设计相结合。为了高效的下游适应，我们提出了一个微调模块，该模块通过社区内和社区间的联合结构学习来优化分层结构。大量实验表明，SA$^{2}$GFM在针对随机噪音和节点和图形分类对抗性扰动的有效性和鲁棒性方面优于9个最先进的基线。



## **30. Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning**

针对两个对手进行调优：使用超参数调优增强针对传输和基于查询的攻击的鲁棒性 cs.LG

To appear in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) 2026

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.13654v2) [paper-pdf](https://arxiv.org/pdf/2511.13654v2)

**Authors**: Pascal Zimmer, Ghassan Karame

**Abstract**: In this paper, we present the first detailed analysis of how training hyperparameters -- such as learning rate, weight decay, momentum, and batch size -- influence robustness against both transfer-based and query-based attacks. Supported by theory and experiments, our study spans a variety of practical deployment settings, including centralized training, ensemble learning, and distributed training. We uncover a striking dichotomy: for transfer-based attacks, decreasing the learning rate significantly enhances robustness by up to $64\%$. In contrast, for query-based attacks, increasing the learning rate consistently leads to improved robustness by up to $28\%$ across various settings and data distributions. Leveraging these findings, we explore -- for the first time -- the training hyperparameter space to jointly enhance robustness against both transfer-based and query-based attacks. Our results reveal that distributed models benefit the most from hyperparameter tuning, achieving a remarkable tradeoff by simultaneously mitigating both attack types more effectively than other training setups.

摘要: 在本文中，我们首次详细分析了训练超参数（如学习率、权重衰减、动量和批量大小）如何影响对基于传输和基于查询的攻击的鲁棒性。在理论和实验的支持下，我们的研究涵盖了各种实际部署环境，包括集中式培训、集成学习和分布式培训。我们发现了一个引人注目的二分法：对于基于传输的攻击，降低学习率可以显着增强鲁棒性，提高高达64美元。相比之下，对于基于查询的攻击，在各种设置和数据分布中持续提高学习率可使稳健性提高高达28美元。利用这些发现，我们首次探索了训练超参数空间，以共同增强针对基于传输和基于查询的攻击的鲁棒性。我们的结果表明，分布式模型从超参数调整中受益最多，通过比其他训练设置更有效地同时减轻两种攻击类型，实现了显着的权衡。



## **31. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

对生成性基因组模型的生物知情混合成员推断攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.07503v3) [paper-pdf](https://arxiv.org/pdf/2511.07503v3)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.

摘要: 遗传数据可用性的增加改变了基因组学研究，但由于其敏感性，对其处理提出了许多隐私问题。这项工作探索了使用语言模型（LM）来生成合成基因突变谱，利用差异隐私（DP）来保护敏感遗传数据。我们通过引入一种新型的生物知情混合成员推断攻击（biHMIA）来经验性地评估DP模式的隐私保证，该攻击将传统的黑匣子MIA与上下文基因组学指标相结合，以增强攻击能力。我们的实验表明，小型和大型Transformer GPT类模型都是小规模基因组学的可行合成变体生成器，并且与传统的基于度量的MIA相比，我们的混合攻击平均会导致更高的对抗成功。



## **32. Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems**

频谱掩蔽和内插攻击（SMIA）：针对语音认证和反欺骗系统的黑匣子对抗攻击 cs.SD

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2509.07677v3) [paper-pdf](https://arxiv.org/pdf/2509.07677v3)

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.

摘要: 语音认证系统（PAS）使用独特的声音特征进行验证。他们越来越多地融入银行和医疗保健等高安全领域。尽管它们使用深度学习进行了改进，但它们仍面临来自深度造假和对抗攻击等复杂威胁的严重漏洞。现实语音克隆的出现使检测变得复杂，因为系统很难区分真实音频和合成音频。虽然存在反欺骗对策（CM）来减轻这些风险，但许多对策依赖于静态检测模型，这些模型可以被新型对抗方法绕过，从而留下了关键的安全漏洞。为了证明这一漏洞，我们提出了频谱掩蔽和内插攻击（SMIA），这是一种新颖的方法，可以战略性地操纵人工智能生成的音频的听不见的频率区域。通过改变人耳不可感知区域的声音，SMIA创建听起来真实的对抗样本，同时欺骗CM。我们在模拟现实世界条件下对多个任务中针对最先进（SOTA）模型的攻击进行了全面评估。SMIA在对抗组合式增值服务器/CM系统时，至少达到82%的攻击成功率（ASB），对抗独立说话者验证系统至少达到97.5%，对抗措施至少达到100%。这些发现最终证明，当前的安全姿态不足以抵御适应性对抗攻击。这项工作凸显了向下一代防御范式转变的迫切需要，这些防御采用能够随着威胁格局而演变的动态、上下文感知框架。



## **33. Larger Scale Offers Better Security in the Nakamoto-style Blockchain**

更大的规模在Nakamoto式区块链中提供更好的安全性 cs.CR

22 pages, 4 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2509.05708v3) [paper-pdf](https://arxiv.org/pdf/2509.05708v3)

**Authors**: Junjie Hu

**Abstract**: Traditional security models for Nakamoto-style blockchains assume instantaneous synchronization among malicious nodes, which overestimate adversarial coordination capability. We revisit these existing models and propose two more realistic security models. First, we propose the static delay model. This model first incorporates adversarial communication delay. It quantifies how the delay constrains the effective growth rate of private chains and yields a closed-form expression for the security threshold. Second, we propose the dynamic delay model that further captures the decay of adversarial corruption capability and the total adversarial delay window. Theoretical analysis shows that private attacks remain optimal under both models. Finally, we prove that large-scale Nakamoto-style blockchains offer better security. This result provided a theoretical foundation for optimizing consensus protocols and assessing the robustness of large-scale blockchains.

摘要: 中本式区块链的传统安全模型假设恶意节点之间的即时同步，这高估了对抗协调能力。我们重新审视这些现有模型并提出两个更现实的安全模型。首先，我们提出了静态延迟模型。该模型首先考虑了对抗性通信延迟。它量化了延迟如何限制私有链的有效增长率，并给出了安全阈值的封闭表达式。其次，我们提出了动态延迟模型，进一步捕捉对抗腐败能力和总对抗延迟窗口的衰减。理论分析表明，在这两种模型下，私有攻击仍然是最优的。最后，我们证明了大规模的Nakamoto式区块链提供了更好的安全性。这一结果为优化共识协议和评估大规模区块链的稳健性提供了理论基础。



## **34. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2508.17361v2) [paper-pdf](https://arxiv.org/pdf/2508.17361v2)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective against basic and reasoning models, but are also transferable across model families (OpenAI, Anthropic, Google), and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅对基本模型和推理模型有效，而且还可以跨模型家族（OpenAI、Anthropic、Google）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **35. Exact Verification of Graph Neural Networks with Incremental Constraint Solving**

增量约束求解的图神经网络精确验证 cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2508.09320v2) [paper-pdf](https://arxiv.org/pdf/2508.09320v2)

**Authors**: Minghao Liu, Chia-Hsuan Lu, Marta Kwiatkowska

**Abstract**: Graph neural networks (GNNs) are increasingly employed in high-stakes applications, such as fraud detection or healthcare, but are susceptible to adversarial attacks. A number of techniques have been proposed to provide adversarial robustness guarantees, but support for commonly used aggregation functions in message-passing GNNs is lacking. In this paper, we develop an exact (sound and complete) verification method for GNNs to compute guarantees against attribute and structural perturbations that involve edge addition or deletion, subject to budget constraints. Our method employs constraint solving with bound tightening, and iteratively solves a sequence of relaxed constraint satisfaction problems while relying on incremental solving capabilities of solvers to improve efficiency. We implement GNNev, a versatile exact verifier for message-passing neural networks, which supports three aggregation functions, sum, max and mean, with the latter two considered here for the first time. Extensive experimental evaluation of GNNev on real-world fraud datasets (Amazon and Yelp) and biochemical datasets (MUTAG and ENZYMES) demonstrates its usability and effectiveness, as well as superior performance for node classification and competitiveness on graph classification compared to existing exact verification tools on sum-aggregated GNNs.

摘要: 图神经网络（GNN）越来越多地用于欺诈检测或医疗保健等高风险应用，但很容易受到对抗攻击。人们提出了多种技术来提供对抗稳健性保证，但缺乏对消息传递GNN中常用的聚合函数的支持。在本文中，我们为GNN开发了一种精确的（合理且完整的）验证方法，以计算针对涉及边添加或删除的属性和结构扰动的保证，并受预算限制。我们的方法采用带边界收紧的约束求解，并迭代地解决一系列宽松的约束满足问题，同时依靠求解器的增量求解能力来提高效率。我们实现了GNNev，这是一个用于消息传递神经网络的通用精确验证器，它支持三个聚合函数：sum、max和mean，其中后两个是本文首次考虑。GNNev在现实世界欺诈数据集（Amazon和Yelp）和生化数据集（MUTAG和ENZYMES）上进行的广泛实验评估证明了其可用性和有效性，以及与现有的精确验证工具相比，在节点分类和图形分类上具有卓越的性能。聚合GNN。



## **36. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

Published in NeurIPS 2025

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2507.06489v3) [paper-pdf](https://arxiv.org/pdf/2507.06489v3)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于LLM的部署至关重要，以帮助确保许多应用程序（包括涉及人机交互的应用程序）的透明度、信任和安全性。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们通过干扰和基于越狱的方法引入了针对言语信心分数的攻击框架，并证明这些攻击会显着损害言语信心估计并导致答案频繁变化。我们检查了各种提示策略、模型大小和应用领域，揭示了当前的言语自信很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了为LLM中的信心表达设计稳健的机制的必要性，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **37. Breaking the Bulkhead: Demystifying Cross-Namespace Reference Vulnerabilities in Kubernetes Operators**

打破障碍：揭开Kubernetes运营商中的跨空间引用漏洞的神秘面纱 cs.CR

18 pages. Accepted by Network and Distributed System Security (NDSS) Symposium 2026. Some information has been omitted from this preprint version due to ethical considerations. The final published version differs from this version

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2507.03387v3) [paper-pdf](https://arxiv.org/pdf/2507.03387v3)

**Authors**: Andong Chen, Ziyi Guo, Zhaoxuan Jin, Zhenyuan Li, Yan Chen

**Abstract**: Kubernetes Operators, automated tools designed to manage application lifecycles within Kubernetes clusters, extend the functionalities of Kubernetes, and reduce the operational burden on human engineers. While Operators significantly simplify DevOps workflows, they introduce new security risks. In particular, Kubernetes enforces namespace isolation to separate workloads and limit user access, ensuring that users can only interact with resources within their authorized namespaces. However, Kubernetes Operators often demand elevated privileges and may interact with resources across multiple namespaces. This introduces a new class of vulnerabilities, the Cross-Namespace Reference Vulnerability. The root cause lies in the mismatch between the declared scope of resources and the implemented scope of the Operator logic, resulting in Kubernetes being unable to properly isolate the namespace. Leveraging such vulnerability, an adversary with limited access to a single authorized namespace may exploit the Operator to perform operations affecting other unauthorized namespaces, causing Privilege Escalation and further impacts.   To the best of our knowledge, this paper is the first to systematically investigate Kubernetes Operator attacks. We present Cross-Namespace Reference Vulnerability with two strategies, demonstrating how an attacker can bypass namespace isolation. Through large-scale measurements, we found that over 14% of Operators in the wild are potentially vulnerable. Our findings have been reported to the relevant developers, resulting in 8 confirmations and 7 CVEs by the time of submission, affecting vendors including Red Hat and NVIDIA, highlighting the critical need for enhanced security practices in Kubernetes Operators. To mitigate it, we open-source the static analysis suite and propose concrete mitigation to benefit the ecosystem.

摘要: Kubernetes Operators是一种自动化工具，旨在管理Kubernetes集群内的应用程序生命周期，扩展Kubernetes的功能，并减少人类工程师的操作负担。虽然运营商大大简化了DevOps工作流程，但它们引入了新的安全风险。特别是，Kubernetes强制执行命名空间隔离，以分离工作负载并限制用户访问，确保用户只能与其授权命名空间内的资源交互。但是，Kubernetes Operator通常需要提升的权限，并且可能会跨多个命名空间与资源交互。这引入了一类新的漏洞，即跨空间引用漏洞。根本原因在于声明的资源范围与Operator逻辑的实现范围不匹配，导致Kubernetes无法正确隔离命名空间。利用此类漏洞，对单个授权命名空间的访问权限有限的对手可能会利用运营商执行影响其他未经授权命名空间的操作，从而导致特权升级和进一步影响。   据我们所知，本文是第一篇系统性研究Kubernetes Operator攻击的论文。我们通过两种策略展示跨命名空间引用漏洞，展示攻击者如何绕过命名空间隔离。通过大规模测量，我们发现超过14%的野外经营者存在潜在的脆弱性。我们的调查结果已报告给相关开发人员，截至提交时已获得8项确认和7项CVS，影响了Red Hat和NVIDIA等供应商，凸显了Kubernetes运营商对增强安全实践的迫切需求。为了缓解这种情况，我们开源了静态分析套件，并提出具体的缓解措施以造福生态系统。



## **38. Benchmarking Gaslighting Negation Attacks Against Reasoning Models**

针对推理模型的煤气灯否定攻击基准 cs.CV

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2506.09677v2) [paper-pdf](https://arxiv.org/pdf/2506.09677v2)

**Authors**: Bin Zhu, Hailong Yin, Jingjing Chen, Yu-Gang Jiang

**Abstract**: Recent advances in reasoning-centric models promise improved robustness through mechanisms such as chain-of-thought prompting and test-time scaling. However, their ability to withstand gaslighting negation attacks-adversarial prompts that confidently deny correct answers-remains underexplored. In this paper, we conduct a systematic evaluation of three state-of-the-art reasoning models, i.e., OpenAI's o4-mini, Claude-3.7-Sonnet and Gemini-2.5-Flash, across three multimodal benchmarks: MMMU, MathVista, and CharXiv. Our evaluation reveals significant accuracy drops (25-29% on average) following gaslighting negation attacks, indicating that even top-tier reasoning models struggle to preserve correct answers under manipulative user feedback. Built upon the insights of the evaluation and to further probe this vulnerability, we introduce GaslightingBench-R, a new diagnostic benchmark specifically designed to evaluate reasoning models' susceptibility to defend their belief under gaslighting negation attacks. Constructed by filtering and curating 1,025 challenging samples from the existing benchmarks, GaslightingBench-R induces even more dramatic failures, with accuracy drops exceeding 53% on average. Our findings highlight a fundamental gap between step-by-step reasoning and resistance to adversarial manipulation, calling for new robustness strategies that safeguard reasoning models against gaslighting negation attacks.

摘要: 以推理为中心的模型的最新进展承诺通过思想链提示和测试时扩展等机制来提高鲁棒性。然而，它们抵御煤气灯否定攻击（自信地否认正确答案的对抗性提示）的能力仍然没有得到充分的探索。在本文中，我们对三种最先进的推理模型进行了系统评估，即OpenAI的o 4-mini、Claude-3.7-Sonnet和Gemini-2.5-Flash，跨三种多模式基准：MMMU、MathVista和CharXiv。我们的评估显示，在煤气灯否定攻击后，准确性显着下降（平均25-29%），这表明即使是顶级推理模型也很难在操纵性用户反馈下保留正确答案。基于评估的见解并进一步探索这一漏洞，我们引入了GaslightingBench-R，这是一种新的诊断基准，专门用于评估推理模型在Gaslightning否定攻击下捍卫其信念的敏感性。GaslightingBench-R通过从现有基准中过滤和整理1，025个具有挑战性的样本而构建，导致了更严重的失败，准确性平均下降超过53%。我们的研究结果强调了逐步推理和对抗性操纵抵抗之间的根本差距，呼吁采取新的稳健性策略来保护推理模型免受煤气灯否定攻击。



## **39. MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models**

MoAPT：视觉语言模型的对抗性提示调优混合 cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2505.17509v2) [paper-pdf](https://arxiv.org/pdf/2505.17509v2)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Maoxun Yuan, Jialing Tao, Jiexi Liu, Ranjie Duan, Jie Zhang, Jie Zhang, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.

摘要: 大型预先训练的视觉语言模型（VLM）表现出出色的概括能力，但仍然极易受到对抗性示例的影响，从而构成潜在的安全风险。为了提高VLM对对抗性示例的鲁棒性，提出了对抗性提示调整方法，以在不改变模型参数的情况下将文本特征与对抗性图像特征对齐。然而，当面临各种对抗性攻击时，单个可学习文本提示的概括性不足以与所有对抗性图像特征很好地对齐，这最终会导致过度匹配。为了解决上述挑战，在本文中，我们经验发现，增加学习提示的数量比简单地延长单个提示的长度可以产生更大的鲁棒性改进。在这一观察的基础上，我们提出了一种名为\textBF{混合对抗提示调整（MoAPT）}的对抗性调整方法，以增强针对VLM的各种对抗性攻击的概括性。MoAPT旨在学习混合文本提示以获得更稳健的文本特征。为了进一步增强适应性，我们提出了一种基于对抗图像的条件权重路由器来预测多个学习提示的混合权重，这有助于获得与不同对抗图像特征对齐的样本特定混合文本特征。在不同设置下对11个数据集进行的广泛实验表明，我们的方法可以实现比最先进的方法更好的对抗鲁棒性。



## **40. FlippedRAG: Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models**

FlippedRAG：黑盒意见操纵对抗性攻击检索增强生成模型 cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2501.02968v5) [paper-pdf](https://arxiv.org/pdf/2501.02968v5)

**Authors**: Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) enriches LLMs by dynamically retrieving external knowledge, reducing hallucinations and satisfying real-time information needs. While existing research mainly targets RAG's performance and efficiency, emerging studies highlight critical security concerns. Yet, current adversarial approaches remain limited, mostly addressing white-box scenarios or heuristic black-box attacks without fully investigating vulnerabilities in the retrieval phase. Additionally, prior works mainly focus on factoid Q&A tasks, their attacks lack complexity and can be easily corrected by advanced LLMs. In this paper, we investigate a more realistic and critical threat scenario: adversarial attacks intended for opinion manipulation against black-box RAG models, particularly on controversial topics. Specifically, we propose FlippedRAG, a transfer-based adversarial attack against black-box RAG systems. We first demonstrate that the underlying retriever of a black-box RAG system can be reverse-engineered, enabling us to train a surrogate retriever. Leveraging the surrogate retriever, we further craft target poisoning triggers, altering vary few documents to effectively manipulate both retrieval and subsequent generation. Extensive empirical results show that FlippedRAG substantially outperforms baseline methods, improving the average attack success rate by 16.7%. FlippedRAG achieves on average a 50% directional shift in the opinion polarity of RAG-generated responses, ultimately causing a notable 20% shift in user cognition. Furthermore, we evaluate the performance of several potential defensive measures, concluding that existing mitigation strategies remain insufficient against such sophisticated manipulation attacks. These results highlight an urgent need for developing innovative defensive solutions to ensure the security and trustworthiness of RAG systems.

摘要: 检索增强生成（RAG）通过动态检索外部知识、减少幻觉和满足实时信息需求来丰富LLM。虽然现有研究主要针对RAG的性能和效率，但新出现的研究强调了关键的安全问题。然而，当前的对抗方法仍然有限，主要解决白盒场景或启发式黑匣子攻击，而没有在检索阶段充分调查漏洞。此外，之前的作品主要集中在事实问答任务上，其攻击缺乏复杂性，并且可以通过高级LLM轻松纠正。在本文中，我们研究了一个更现实和关键的威胁场景：针对黑箱RAG模型的意见操纵的对抗性攻击，特别是在有争议的话题上。具体来说，我们提出了FlippedRAG，这是一种针对黑盒RAG系统的基于传输的对抗性攻击。我们首先证明了一个黑盒RAG系统的底层检索器可以进行逆向工程，使我们能够训练一个代理检索器。利用代理检索器，我们进一步工艺目标中毒触发器，改变不同的几个文件，以有效地操纵检索和后续生成。广泛的实证结果表明，FlippedRAG的性能大大优于基线方法，将平均攻击成功率提高了16.7%。FlippedRAG平均实现了RAG生成的响应的意见两极50%的方向性转变，最终导致用户认知发生了20%的显着转变。此外，我们评估了几种潜在防御措施的性能，得出的结论是，现有的缓解策略仍然不足以应对此类复杂的操纵攻击。这些结果凸显了开发创新防御解决方案的迫切需要，以确保RAG系统的安全性和可信性。



## **41. Improving Graph Neural Network Training, Defense and Hypergraph Partitioning via Adversarial Robustness Evaluation**

通过对抗稳健性评估改进图神经网络训练、防御和超图划分 cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2412.14738v9) [paper-pdf](https://arxiv.org/pdf/2412.14738v9)

**Authors**: Yongyu Wang

**Abstract**: Graph Neural Networks (GNNs) are a highly effective neural network architecture for processing graph-structured data. Unlike traditional neural networks that rely solely on the features of the data as input, GNNs leverage both the graph structure, which represents the relationships between data points, and the feature matrix of the data to optimize their feature representation. This unique capability enables GNNs to achieve superior performance across various tasks. However, it also makes GNNs more susceptible to noise and adversarial attacks from both the graph structure and data features, which can significantly increase the training difficulty and degrade their performance. Similarly, a hypergraph is a highly complex structure, and partitioning a hypergraph is a challenging task. This paper leverages spectral adversarial robustness evaluation to effectively address key challenges in complex-graph algorithms. By using spectral adversarial robustness evaluation to distinguish robust nodes from non-robust ones and treating them differently, we propose a training-set construction strategy that improves the training quality of GNNs. In addition, we develop algorithms to enhance both the adversarial robustness of GNNs and the performance of hypergraph partitioning. Experimental results show that this series of methods is highly effective.

摘要: 图神经网络（GNN）是一种用于处理图结构数据的高效神经网络架构。与仅依赖数据特征作为输入的传统神经网络不同，GNN利用表示数据点之间关系的图结构和数据的特征矩阵来优化其特征表示。这种独特的功能使GNN能够在各种任务中实现卓越的性能。然而，它也使GNN更容易受到来自图结构和数据特征的噪音和对抗攻击，这可能会显着增加训练难度并降低其性能。同样，超图是一种高度复杂的结构，划分超图是一项具有挑战性的任务。本文利用谱对抗鲁棒性评估来有效解决复杂图算法中的关键挑战。通过使用谱对抗鲁棒性评估来区分鲁棒节点和非鲁棒节点并区别对待它们，我们提出了一种提高GNN训练质量的训练集构建策略。此外，我们还开发了算法来增强GNN的对抗鲁棒性和超图分区的性能。实验结果表明，该系列方法非常有效。



## **42. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2411.16782v4) [paper-pdf](https://arxiv.org/pdf/2411.16782v4)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Jun Zhu, Yinpeng Dong

**Abstract**: Adversarial examples exhibit cross-model transferability, enabling threatening black-box attacks on commercial models. Model ensembling, which attacks multiple surrogate models, is a known strategy to improve this transferability. However, prior studies typically use small, fixed ensembles, which leaves open an intriguing question of whether scaling the number of surrogate models can further improve black-box attacks. In this work, we conduct the first large-scale empirical study of this question. We show that by resolving gradient conflict with advanced optimizers, we discover a robust and universal log-linear scaling law through both theoretical analysis and empirical evaluations: the Attack Success Rate (ASR) scales linearly with the logarithm of the ensemble size $T$. We rigorously verify this law across standard classifiers, SOTA defenses, and MLLMs, and find that scaling distills robust, semantic features of the target class. Consequently, we apply this fundamental insight to benchmark SOTA MLLMs. This reveals both the attack's devastating power and a clear robustness hierarchy: we achieve 80\%+ transfer attack success rate on proprietary models like GPT-4o, while also highlighting the exceptional resilience of Claude-3.5-Sonnet. Our findings urge a shift in focus for robustness evaluation: from designing intricate algorithms on small ensembles to understanding the principled and powerful threat of scaling.

摘要: 对抗性示例表现出跨模型的可移植性，从而能够对商业模型进行威胁性的黑匣子攻击。攻击多个代理模型的模型集成是提高这种可移植性的已知策略。然而，之前的研究通常使用小型、固定的集合，这留下了一个有趣的问题：扩大代理模型的数量是否可以进一步改善黑匣子攻击。在这项工作中，我们对这个问题进行了首次大规模的实证研究。我们表明，通过使用高级优化器解决梯度冲突，我们通过理论分析和经验评估发现了一个鲁棒且通用的日志线性缩放定律：攻击成功率（ASB）与总体大小$T$的log呈线性缩放。我们在标准分类器、SOTA防御和MLLM之间严格验证了这一定律，并发现缩放可以提炼出目标类的稳健语义特征。因此，我们将这一基本见解应用于基准SOTA MLLM。这既揭示了攻击的破坏力，又揭示了明确的鲁棒性层次结构：我们在GPT-4 o等专有模型上实现了80%以上的传输攻击成功率，同时也凸显了Claude-3.5-Sonnet的非凡弹性。我们的研究结果促使稳健性评估的重点发生转变：从在小集合上设计复杂的算法到理解扩展的原则性且强大的威胁。



## **43. Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors**

针对车辆检测器实现稳健、准确的对抗性伪装生成 cs.CV

14 pages. arXiv admin note: substantial text overlap with arXiv:2402.15853

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2411.10029v2) [paper-pdf](https://arxiv.org/pdf/2411.10029v2)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, End-to-End Neural Renderer Plus (E2E-NRP), which can accurately optimize and project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings.

摘要: 对抗伪装因其在多视角攻击性能上的优势而被广泛使用，是针对车辆探测器的物理攻击。一种有前途的方法涉及使用可微神经渲染器通过梯度反向传播促进对抗伪装优化。然而，现有方法通常难以在渲染过程中捕捉环境特征或产生可以精确映射到目标车辆的对抗纹理。此外，这些方法忽视了不同的天气条件，降低了在不同天气场景下生成的伪装的功效。为了应对这些挑战，我们提出了一种稳健且准确的伪装生成方法，即RAUCA。RAUCA的核心是一个新型神经渲染组件端到端神经渲染Plus（E2 E-NRP），它可以准确地优化和投影车辆纹理，并渲染具有灯光和天气等环境特征的图像。此外，我们还集成了多天气数据集来生成伪装，利用E2 E-NRP来增强攻击稳健性。六种流行物体检测器的实验结果表明，RAUCA-final在模拟和现实环境中都优于现有方法。



## **44. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

人工智能生成图像检测中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2407.20836v5) [paper-pdf](https://arxiv.org/pdf/2407.20836v5)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g., transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we demonstrate that adversarial attacks pose a real threat to AIGI detectors. FPBA can deliver successful black-box attacks across various detectors, generators, defense methods, and even evade cross-generator and compressed image detection, which are crucial real-world detection scenarios. Our code is available at https://github.com/onotoa/fpba.

摘要: 图像合成领域的最新进展，特别是GAN和扩散模型的出现，加剧了公众对虚假信息传播的担忧。为了解决这些问题，人们提出了许多人工智能生成的图像（AIGI）检测器，并在识别虚假图像方面取得了良好的性能。然而，人们仍然缺乏对AIGI检测器对抗鲁棒性的系统了解。在本文中，我们研究了最先进的AIGI检测器在白盒和黑盒设置下对抗攻击的脆弱性，迄今为止，这很少被研究。为此，我们提出了一种攻击AIGI检测器的新方法。首先，受到频域中真实图像和假图像之间明显差异的启发，我们在频域下添加扰动，以推动图像远离其原始频率分布。其次，我们探索代理模型的完整后验分布，以进一步缩小异类AIGI检测器之间的差距，例如，跨CNN和ViT传输对抗性示例。这是通过引入一种新型的训练后Bayesian策略来实现的，该策略将单个代理变成了Bayesian代理，能够使用一个预先训练的代理来模拟不同的受害者模型，而无需重新训练。我们将我们的方法命名为基于频率的训练后Bayesian Attack（FPBA）。通过FPBA，我们证明了对抗性攻击对AIGI检测器构成了真正的威胁。FPBA可以跨各种检测器、生成器、防御方法提供成功的黑匣子攻击，甚至规避交叉生成器和压缩图像检测，这些都是现实世界中至关重要的检测场景。我们的代码可在https://github.com/onotoa/fpba上获取。



## **45. Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis**

神经网络中的过度参数化和对抗鲁棒性：概述和实证分析 cs.LG

Submitted to Discover AI

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2406.10090v2) [paper-pdf](https://arxiv.org/pdf/2406.10090v2)

**Authors**: Srishti Gupta, Zhang Chen, Luca Demetrio, Xiaoyi Feng, Zhaoqiang Xia, Antonio Emanuele Cinà, Maura Pintor, Luca Oneto, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Thanks to their extensive capacity, over-parameterized neural networks exhibit superior predictive capabilities and generalization. However, having a large parameter space is considered one of the main suspects of the neural networks' vulnerability to adversarial example -- input samples crafted ad-hoc to induce a desired misclassification. Relevant literature has claimed contradictory remarks in support of and against the robustness of over-parameterized networks. These contradictory findings might be due to the failure of the attack employed to evaluate the networks' robustness. Previous research has demonstrated that depending on the considered model, the algorithm employed to generate adversarial examples may not function properly, leading to overestimating the model's robustness. In this work, we empirically study the robustness of over-parameterized networks against adversarial examples. However, unlike the previous works, we also evaluate the considered attack's reliability to support the results' veracity. Our results show that over-parameterized networks are robust against adversarial attacks as opposed to their under-parameterized counterparts.

摘要: 由于其广泛的容量，过度参数化神经网络展现出卓越的预测能力和概括性。然而，拥有大的参数空间被认为是神经网络容易受到对抗性示例影响的主要嫌疑人之一--即专门制作的输入样本，以引发所需的错误分类。相关文献在支持和反对过度参数化网络的鲁棒性方面提出了相互矛盾的言论。这些相互矛盾的发现可能是由于用于评估网络稳健性的攻击失败所致。之前的研究表明，根据所考虑的模型，用于生成对抗性示例的算法可能无法正常运行，从而导致高估模型的稳健性。在这项工作中，我们实证研究了过度参数化网络对对抗示例的鲁棒性。然而，与之前的作品不同的是，我们还评估了所考虑的攻击的可靠性，以支持结果的准确性。我们的结果表明，与参数化不足的网络相比，过度参数化的网络对对抗攻击具有鲁棒性。



## **46. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

MIIR：基于互信息的对抗鲁棒性的掩蔽图像建模 cs.CV

Accepted by NDSS 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2312.04960v5) [paper-pdf](https://arxiv.org/pdf/2312.04960v5)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) have emerged as a fundamental architecture and serve as the backbone of modern vision-language models. Despite their impressive performance, ViTs exhibit notable vulnerability to evasion attacks, necessitating the development of specialized Adversarial Training (AT) strategies tailored to their unique architecture. While a direct solution might involve applying existing AT methods to ViTs, our analysis reveals significant incompatibilities, particularly with state-of-the-art (SOTA) approaches such as Generalist (CVPR 2023) and DBAT (USENIX Security 2024). This paper presents a systematic investigation of adversarial robustness in ViTs and provides a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained via derived MI bounds. Building on this insight, we propose a self-supervised AT method, MIMIR, that employs an MI penalty to facilitate adversarial pre-training by masked image modeling with autoencoders. Extensive experiments on CIFAR-10, Tiny-ImageNet, and ImageNet-1K show that MIMIR can consistently provide improved natural and robust accuracy, where MIMIR outperforms SOTA AT results on ImageNet-1K. Notably, MIMIR demonstrates superior robustness against unforeseen attacks and common corruption data and can also withstand adaptive attacks where the adversary possesses full knowledge of the defense mechanism. Our code and trained models are publicly available at: https://github.com/xiaoyunxxy/MIMIR.

摘要: 视觉变形者（ViT）已成为一种基本架构，并成为现代视觉语言模型的支柱。尽管ViT的性能令人印象深刻，但其对规避攻击表现出明显的脆弱性，因此需要开发针对其独特架构定制的专门对抗训练（AT）策略。虽然直接的解决方案可能涉及将现有的AT方法应用于ViT，但我们的分析揭示了显着的不兼容性，特别是与最先进的（SOTA）方法，例如Generalist（CVPR 2023）和DBAT（USENIX Security 2024）。本文对ViT中的对抗鲁棒性进行了系统研究，并在其基于自动编码器的自我监督预训练中提供了一种新颖的理论互信息（MI）分析。具体来说，我们表明对抗性示例及其在基于ViT的自动编码器中的潜在表示之间的MI应该通过推导出的MI界限来约束。基于这一见解，我们提出了一种自我监督的AT方法MIIR，它采用MI罚分来通过使用自动编码器进行掩蔽图像建模来促进对抗性预训练。CIFAR-10、Tiny-ImageNet和ImageNet-1 K上的大量实验表明，MIIR可以始终如一地提供改进的自然和稳健的准确性，其中MIIR优于ImageNet-1 K上的SOTA AT结果。值得注意的是，MIIR表现出针对不可预见的攻击和常见腐败数据的卓越鲁棒性，并且还可以抵御对手完全了解防御机制的自适应攻击。我们的代码和训练模型可在以下网址公开获取：https://github.com/xiaoyunxxy/MIMIR。



## **47. Enigma: Application-Layer Privacy for Quantum Optimization on Untrusted Computers**

Enigma：不可信计算机上量子优化的应用层隐私 quant-ph

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2311.13546v2) [paper-pdf](https://arxiv.org/pdf/2311.13546v2)

**Authors**: Ramin Ayanzadeh, Ahmad Mousavi, Amirhossein Basareh, Narges Alavisamani, Kazem Taram

**Abstract**: The Early Fault-Tolerant (EFT) era is emerging, where modest Quantum Error Correction (QEC) can enable quantum utility before full-scale fault tolerance. Quantum optimization is a leading candidate for early applications, but protecting these workloads is critical since they will run on expensive cloud services where providers could learn sensitive problem details. Experience with classical computing systems has shown that treating security as an afterthought can lead to significant vulnerabilities. Thus, we must address the security implications of quantum computing before widespread adoption. However, current Secure Quantum Computing (SQC) approaches, although theoretically promising, are impractical in the EFT era: blind quantum computing requires large-scale quantum networks, and quantum homomorphic encryption depends on full QEC.   We propose application-specific SQC, a principle that applies obfuscation at the application layer to enable practical deployment while remaining agnostic to algorithms, computing models, and hardware architectures. We present Enigma, the first realization of this principle for quantum optimization. Enigma integrates three complementary obfuscations: ValueGuard scrambles coefficients, StructureCamouflage inserts decoys, and TopologyTrimmer prunes variables. These techniques guarantee recovery of original solutions, and their stochastic nature resists repository-matching attacks. Evaluated against seven state-of-the-art AI models across five representative graph families, even combined adversaries, under a conservatively strong attacker model, identify the correct problem within their top five guesses in only 4.4% of cases. The protections come at the cost of problem size and T-gate counts increasing by averages of 1.07x and 1.13x, respectively, with both obfuscation and decoding completing within seconds for large-scale problems.

摘要: 早期故障容忍（EFT）时代正在兴起，适度的量子错误纠正（QEC）可以在全面故障容忍之前实现量子效用。量子优化是早期应用程序的主要候选者，但保护这些工作负载至关重要，因为它们将在昂贵的云服务上运行，提供商可以了解敏感的问题详细信息。经典计算系统的经验表明，将安全性视为事后考虑可能会导致重大漏洞。因此，我们必须在广泛采用量子计算之前解决量子计算的安全影响。然而，当前的安全量子计算（SQC）方法尽管在理论上很有希望，但在EFT时代是不切实际的：盲量子计算需要大规模量子网络，而量子同质加密依赖于完整的QEC。   我们提出了特定于应用程序的SQC，这一原则在应用程序层应用模糊处理，以实现实际部署，同时对算法、计算模型和硬件架构保持不可知。我们提出Enigma，这是量子优化原理的第一次实现。Enigma集成了三种互补的混淆：ValueGuard扰乱系数、StructureCamerage插入诱饵以及TopologyTrimmer修剪变量。这些技术保证原始解决方案的恢复，并且其随机性可以抵抗存储库匹配攻击。在保守强大的攻击者模型下，针对五个代表性图族的七个最先进的人工智能模型进行评估，即使是组合的对手，也只能在4.4%的情况下在前五个猜测中识别出正确的问题。这些保护的代价是问题大小和T门计数平均分别增加1.07倍和1.13倍，对于大规模问题，混淆和解码都在几秒钟内完成。



## **48. Why Does Little Robustness Help? A Further Step Towards Understanding Adversarial Transferability**

为什么小鲁棒性有帮助？了解对抗性可转让性的又一步 cs.LG

IEEE Symposium on Security and Privacy (Oakland) 2024; Extended version; Fix an proof error of Theorem 1

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2307.07873v8) [paper-pdf](https://arxiv.org/pdf/2307.07873v8)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.

摘要: DNN的对抗示例（AE）已经被证明是可转移的：成功欺骗白盒代理模型的AE也可以欺骗其他具有不同架构的黑盒模型。尽管大量的实证研究为产生高度可转移的不良事件提供了指导，但其中许多研究结果缺乏解释，甚至导致不一致的建议。在本文中，我们采取了进一步的理解对抗性的可转让性，特别侧重于代理方面。从有趣的小鲁棒性现象开始，其中使用轻度扰动的对抗样本进行对抗训练的模型可以作为更好的替代品，我们将其归因于两个主要因素之间的权衡：模型平滑度和梯度相似性。我们的研究重点是它们的联合影响，而不是它们与可转移性的单独相关性。通过一系列理论和实证分析，我们推测对抗训练中的数据分布变化解释了梯度相似性的下降。在这些见解的基础上，我们探索了数据增强和梯度正规化对可移植性的影响，并确定各种训练机制中普遍存在权衡，从而为可移植性背后的监管机制构建了全面的蓝图。最后，我们提供了一种通用路线来构建更好的代理以提高可移植性，该路线同时优化模型平滑度和梯度相似性，例如，输入梯度正规化和清晰度感知最小化（Sam）的结合，经过大量实验验证。总而言之，我们呼吁关注这两个因素对发起有效传输攻击的联合影响，而不是优化其中一个因素而忽略另一个因素，并强调操纵代理模型的关键作用。



## **49. A First Order Meta Stackelberg Method for Robust Federated Learning (Technical Report)**

鲁棒联邦学习的一阶Meta Stackelberg方法（技术报告） cs.CR

This submission is a technical report for "A First Order Meta Stackelberg Method for Robust Federated Learning" (arXiv:2306.13800). We later submitted a full paper, "Meta Stackelberg Game: Robust Federated Learning Against Adaptive and Mixed Poisoning Attacks" (arXiv:2410.17431), which fully incorporates this report in its Appendix. To avoid duplication, we withdraw this submission

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2306.13273v3) [paper-pdf](https://arxiv.org/pdf/2306.13273v3)

**Authors**: Henger Li, Tianyi Xu, Tao Li, Yunian Pan, Quanyan Zhu, Zizhan Zheng

**Abstract**: Recent research efforts indicate that federated learning (FL) systems are vulnerable to a variety of security breaches. While numerous defense strategies have been suggested, they are mainly designed to counter specific attack patterns and lack adaptability, rendering them less effective when facing uncertain or adaptive threats. This work models adversarial FL as a Bayesian Stackelberg Markov game (BSMG) between the defender and the attacker to address the lack of adaptability to uncertain adaptive attacks. We further devise an effective meta-learning technique to solve for the Stackelberg equilibrium, leading to a resilient and adaptable defense. The experiment results suggest that our meta-Stackelberg learning approach excels in combating intense model poisoning and backdoor attacks of indeterminate types.

摘要: 最近的研究工作表明，联邦学习（FL）系统容易受到各种安全漏洞的影响。虽然已经提出了许多防御策略，但它们主要是为了对抗特定的攻击模式而设计的，并且缺乏适应性，导致它们在面临不确定或适应性威胁时效果较差。这项工作将对抗FL建模为防御者和攻击者之间的Bayesian Stackelberg Markov博弈（BSMG），以解决对不确定的适应性攻击缺乏适应性的问题。我们进一步设计了一种有效的元学习技术来解决Stackelberg均衡，从而实现弹性和适应性强的防御。实验结果表明，我们的元Stackelberg学习方法在对抗不确定类型的强烈模型中毒和后门攻击方面表现出色。



## **50. We Can Always Catch You: Detecting Adversarial Patched Objects WITH or WITHOUT Signature**

我们总是可以抓住你：检测带有或不带有签名的敌对修补对象 cs.CV

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2106.05261v3) [paper-pdf](https://arxiv.org/pdf/2106.05261v3)

**Authors**: Jiachun Li, Jianan Feng, Jianjun Huang, Bin Liang

**Abstract**: Recently, object detection has proven vulnerable to adversarial patch attacks. The attackers holding a specially crafted patch can hide themselves from state-of-the-art detectors, e.g., YOLO, even in the physical world. This attack can bring serious security threats, such as escaping from surveillance cameras. How to effectively detect this kind of adversarial examples to catch potential attacks has become an important problem. In this paper, we propose two detection methods: the signature-based method and the signature-independent method. First, we identify two signatures of existing adversarial patches that can be utilized to precisely locate patches within adversarial examples. By employing the signatures, a fast signature-based method is developed to detect the adversarial objects. Second, we present a robust signature-independent method based on the \textit{content semantics consistency} of model outputs. Adversarial objects violate this consistency, appearing locally but disappearing globally, while benign ones remain consistently present. The experiments demonstrate that two proposed methods can effectively detect attacks both in the digital and physical world. These methods each offer distinct advantage. Specifically, the signature-based method is capable of real-time detection, while the signature-independent method can detect unknown adversarial patch attacks and makes defense-aware attacks almost impossible to perform.

摘要: 最近，对象检测已被证明容易受到对抗补丁攻击。持有特制补丁的攻击者可以隐藏自己，免受最先进的检测器的侵害，例如，YOLO，即使在物理世界中也是如此。这种攻击可能会带来严重的安全威胁，例如逃离监控摄像头。如何有效检测此类对抗性示例以捕捉潜在的攻击已成为一个重要问题。本文提出了两种检测方法：基于签名的方法和独立签名的方法。首先，我们识别现有对抗性补丁的两个签名，可用于在对抗性示例中准确定位补丁。通过利用这些签名，开发了一种基于签名的快速方法来检测对抗对象。其次，我们提出了一种基于模型输出的\textit{内容语义一致性}的稳健签名无关方法。敌对对象违反了这种一致性，出现在局部但在全球范围内消失，而良性对象则始终存在。实验表明，提出的两种方法可以有效检测数字和物理世界中的攻击。这些方法都具有独特的优势。具体来说，基于签名的方法能够实时检测，而独立签名的方法可以检测未知的对抗性补丁攻击，并使防御感知攻击几乎不可能执行。



