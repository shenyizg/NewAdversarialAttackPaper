# Latest Adversarial Attack Papers
**update at 2025-10-20 09:09:18**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Towards Proactive Defense Against Cyber Cognitive Attacks**

主动防御网络认知攻击 cs.CR

University of Colorado Colorado Springs and Department of the Air  Force, US Air Force Academy. Disclaimer: The views expressed are those of the  author and do not reflect the official policy or position of the US Air Force  Academy, US Air Force, Department of Defense, or the US Government

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15801v1) [paper-pdf](http://arxiv.org/pdf/2510.15801v1)

**Authors**: Bonnie Rushing, Mac-Rufus Umeokolo, Shouhuai Xu

**Abstract**: Cyber cognitive attacks leverage disruptive innovations (DIs) to exploit psychological biases and manipulate decision-making processes. Emerging technologies, such as AI-driven disinformation and synthetic media, have accelerated the scale and sophistication of these threats. Prior studies primarily categorize current cognitive attack tactics, lacking predictive mechanisms to anticipate future DIs and their malicious use in cognitive attacks. This paper addresses these gaps by introducing a novel predictive methodology for forecasting the emergence of DIs and their malicious uses in cognitive attacks. We identify trends in adversarial tactics and propose proactive defense strategies.

摘要: 网络认知攻击利用破坏性创新（DI）来利用心理偏见并操纵决策过程。人工智能驱动的虚假信息和合成媒体等新兴技术加速了这些威胁的规模和复杂性。之前的研究主要对当前的认知攻击策略进行分类，缺乏预测机制来预测未来的DI及其在认知攻击中的恶意使用。本文通过引入一种新颖的预测方法来预测DI的出现及其在认知攻击中的恶意使用来解决这些差距。我们识别对抗策略的趋势并提出积极主动的防御策略。



## **2. Constrained Adversarial Perturbation**

约束对抗性扰动 cs.LG

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15699v1) [paper-pdf](http://arxiv.org/pdf/2510.15699v1)

**Authors**: Virendra Nishad, Bhaskar Mukhoty, Hilal AlQuabeh, Sandeep K. Shukla, Sayak Ray Chowdhury

**Abstract**: Deep neural networks have achieved remarkable success in a wide range of classification tasks. However, they remain highly susceptible to adversarial examples - inputs that are subtly perturbed to induce misclassification while appearing unchanged to humans. Among various attack strategies, Universal Adversarial Perturbations (UAPs) have emerged as a powerful tool for both stress testing model robustness and facilitating scalable adversarial training. Despite their effectiveness, most existing UAP methods neglect domain specific constraints that govern feature relationships. Violating such constraints, such as debt to income ratios in credit scoring or packet flow invariants in network communication, can render adversarial examples implausible or easily detectable, thereby limiting their real world applicability.   In this work, we advance universal adversarial attacks to constrained feature spaces by formulating an augmented Lagrangian based min max optimization problem that enforces multiple, potentially complex constraints of varying importance. We propose Constrained Adversarial Perturbation (CAP), an efficient algorithm that solves this problem using a gradient based alternating optimization strategy. We evaluate CAP across diverse domains including finance, IT networks, and cyber physical systems, and demonstrate that it achieves higher attack success rates while significantly reducing runtime compared to existing baselines. Our approach also generalizes seamlessly to individual adversarial perturbations, where we observe similar strong performance gains. Finally, we introduce a principled procedure for learning feature constraints directly from data, enabling broad applicability across domains with structured input spaces.

摘要: 深度神经网络在广泛的分类任务中取得了显着的成功。然而，它们仍然非常容易受到对抗性例子的影响--这些输入被微妙地扰动以导致错误分类，而对人类来说似乎没有变化。在各种攻击策略中，通用对抗性扰动（UPC）已成为压力测试模型稳健性和促进可扩展对抗性训练的强大工具。尽管它们有效，但大多数现有的UAP方法都忽视了管理特征关系的特定领域约束。违反此类约束，例如信用评分中的债务与收入比或网络通信中的包流不变量，可能会使对抗性示例变得难以置信或容易检测，从而限制其现实世界的适用性。   在这项工作中，我们通过制定基于增广拉格朗日的最小最大优化问题来推进对受约束特征空间的普遍对抗攻击，该问题强制执行多个重要性不同的潜在复杂约束。我们提出了约束对抗扰动（CAP），这是一种使用基于梯度的交替优化策略来解决这个问题的有效算法。我们评估了金融、IT网络和网络物理系统等多个领域的CAP，并证明与现有基线相比，它可以实现更高的攻击成功率，同时显着缩短运行时间。我们的方法还无缝地推广到个体对抗性扰动，其中我们观察到类似的强劲性能收益。最后，我们引入了一个直接从数据学习特征约束的原则性过程，从而实现跨具有结构化输入空间的领域的广泛适用性。



## **3. Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review**

检测人工智能生成图像的方法和趋势：全面回顾 cs.CV

34 pages, 4 Figures, 10 Tables

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2502.15176v2) [paper-pdf](http://arxiv.org/pdf/2502.15176v2)

**Authors**: Arpan Mahara, Naphtali Rishe

**Abstract**: The proliferation of generative models, such as Generative Adversarial Networks (GANs), Diffusion Models, and Variational Autoencoders (VAEs), has enabled the synthesis of high-quality multimedia data. However, these advancements have also raised significant concerns regarding adversarial attacks, unethical usage, and societal harm. Recognizing these challenges, researchers have increasingly focused on developing methodologies to detect synthesized data effectively, aiming to mitigate potential risks. Prior reviews have predominantly focused on deepfake detection and often overlook recent advancements in synthetic image forensics, particularly approaches that incorporate multimodal frameworks, reasoning-based detection, and training-free methodologies. To bridge this gap, this survey provides a comprehensive and up-to-date review of state-of-the-art techniques for detecting and classifying synthetic images generated by advanced generative AI models. The review systematically examines core detection paradigms, categorizes them into spatial-domain, frequency-domain, fingerprint-based, patch-based, training-free, and multimodal reasoning-based frameworks, and offers concise descriptions of their underlying principles. We further provide detailed comparative analyses of these methods on publicly available datasets to assess their generalizability, robustness, and interpretability. Finally, the survey highlights open challenges and future directions, emphasizing the potential of hybrid frameworks that combine the efficiency of training-free approaches with the semantic reasoning of multimodal models to advance trustworthy and explainable synthetic image forensics.

摘要: 生成模型（例如生成对抗网络（GAN）、扩散模型和变分自动编码器（VAE））的激增使得高质量多媒体数据的合成成为可能。然而，这些进步也引发了人们对对抗性攻击、不道德使用和社会危害的严重担忧。认识到这些挑战，研究人员越来越专注于开发有效检测合成数据的方法，旨在降低潜在风险。之前的评论主要集中在深度伪造检测上，并且经常忽视合成图像取证的最新进展，特别是结合多模式框架、基于推理的检测和免训练方法的方法。为了弥合这一差距，这项调查对用于检测和分类由先进生成式人工智能模型生成的合成图像的最新技术进行了全面且最新的审查。该评论系统地检查了核心检测范式，将它们分为空间域、频域、基于指纹、基于补丁、无训练和基于多模式推理的框架，并对其基本原理进行了简洁的描述。我们进一步在公开可用的数据集上对这些方法进行详细的比较分析，以评估它们的通用性、稳健性和可解释性。最后，该调查强调了开放的挑战和未来方向，强调了混合框架的潜力，该框架将免训练方法的效率与多模式模型的语义推理相结合，以推进值得信赖和可解释的合成图像取证。



## **4. Backdoor or Manipulation? Graph Mixture of Experts Can Defend Against Various Graph Adversarial Attacks**

后门还是操纵？图形混合专家可以防御各种图形对抗攻击 cs.LG

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15333v1) [paper-pdf](http://arxiv.org/pdf/2510.15333v1)

**Authors**: Yuyuan Feng, Bin Ma, Enyan Dai

**Abstract**: Extensive research has highlighted the vulnerability of graph neural networks (GNNs) to adversarial attacks, including manipulation, node injection, and the recently emerging threat of backdoor attacks. However, existing defenses typically focus on a single type of attack, lacking a unified approach to simultaneously defend against multiple threats. In this work, we leverage the flexibility of the Mixture of Experts (MoE) architecture to design a scalable and unified framework for defending against backdoor, edge manipulation, and node injection attacks. Specifically, we propose an MI-based logic diversity loss to encourage individual experts to focus on distinct neighborhood structures in their decision processes, thus ensuring a sufficient subset of experts remains unaffected under perturbations in local structures. Moreover, we introduce a robustness-aware router that identifies perturbation patterns and adaptively routes perturbed nodes to corresponding robust experts. Extensive experiments conducted under various adversarial settings demonstrate that our method consistently achieves superior robustness against multiple graph adversarial attacks.

摘要: 广泛的研究强调了图神经网络（GNN）对对抗攻击的脆弱性，包括操纵、节点注入和最近出现的后门攻击威胁。然而，现有的防御通常专注于单一类型的攻击，缺乏同时防御多种威胁的统一方法。在这项工作中，我们利用混合专家（MoE）架构的灵活性来设计一个可扩展且统一的框架，用于防御后门、边缘操纵和节点注入攻击。具体来说，我们提出了基于MI的逻辑多样性损失，以鼓励各个专家在其决策过程中专注于不同的邻居结构，从而确保足够的专家子集在局部结构的扰动下不受影响。此外，我们引入了一个鲁棒性感知路由器，它识别扰动模式并自适应地将受扰动的节点路由到相应的鲁棒专家。在各种对抗设置下进行的大量实验表明，我们的方法始终能够针对多个图对抗攻击实现卓越的鲁棒性。



## **5. DSSmoothing: Toward Certified Dataset Ownership Verification for Pre-trained Language Models via Dual-Space Smoothing**

DSA平滑：通过双空间平滑实现预训练语言模型的认证数据集所有权验证 cs.CR

13 pages, 21 figures

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15303v1) [paper-pdf](http://arxiv.org/pdf/2510.15303v1)

**Authors**: Ting Qiao, Xing Liu, Wenke Huang, Jianbin Li, Zhaoxin Fan, Yiming Li

**Abstract**: Large web-scale datasets have driven the rapid advancement of pre-trained language models (PLMs), but unauthorized data usage has raised serious copyright concerns. Existing dataset ownership verification (DOV) methods typically assume that watermarks remain stable during inference; however, this assumption often fails under natural noise and adversary-crafted perturbations. We propose the first certified dataset ownership verification method for PLMs based on dual-space smoothing (i.e., DSSmoothing). To address the challenges of text discreteness and semantic sensitivity, DSSmoothing introduces continuous perturbations in the embedding space to capture semantic robustness and applies controlled token reordering in the permutation space to capture sequential robustness. DSSmoothing consists of two stages: in the first stage, triggers are collaboratively embedded in both spaces to generate norm-constrained and robust watermarked datasets; in the second stage, randomized smoothing is applied in both spaces during verification to compute the watermark robustness (WR) of suspicious models and statistically compare it with the principal probability (PP) values of a set of benign models. Theoretically, DSSmoothing provides provable robustness guarantees for dataset ownership verification by ensuring that WR consistently exceeds PP under bounded dual-space perturbations. Extensive experiments on multiple representative web datasets demonstrate that DSSmoothing achieves stable and reliable verification performance and exhibits robustness against potential adaptive attacks.

摘要: 大型网络规模的数据集推动了预训练语言模型（PLM）的快速发展，但未经授权的数据使用引发了严重的版权问题。现有的数据集所有权验证（DOV）方法通常假设水印在推理过程中保持稳定;然而，这种假设在自然噪声和恶意干扰下往往会失败。我们提出了第一个基于双空间平滑的PLM认证数据集所有权验证方法（即，DSS平滑）。为了解决文本离散性和语义敏感性的挑战，DSSmooghing在嵌入空间中引入连续扰动以捕获语义稳健性，并在排列空间中应用受控令牌重新排序以捕获序列稳健性。DSA平滑由两个阶段组成：在第一阶段，触发器协作嵌入两个空间中，以生成受规范约束且稳健的水印数据集;在第二阶段，在验证期间在两个空间中应用随机平滑，以计算可疑模型的水印稳健性（WR），并将其与一组良性模型的主概率（PP）值进行统计比较。理论上，DSSmooting通过确保WR在有界双空间扰动下始终超过PP，为数据集所有权验证提供了可证明的稳健性保证。对多个代表性Web数据集的广泛实验表明，DSSmooghing实现了稳定可靠的验证性能，并对潜在的自适应攻击表现出鲁棒性。



## **6. Keep Calm and Avoid Harmful Content: Concept Alignment and Latent Manipulation Towards Safer Answers**

保持冷静并避免有害内容：概念一致和潜在操纵以获得更安全的答案 cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.12672v2) [paper-pdf](http://arxiv.org/pdf/2510.12672v2)

**Authors**: Ruben Belo, Marta Guimaraes, Claudia Soares

**Abstract**: Large Language Models are susceptible to jailbreak attacks that bypass built-in safety guardrails (e.g., by tricking the model with adversarial prompts). We propose Concept Alignment and Concept Manipulation CALM, an inference-time method that suppresses harmful concepts by modifying latent representations of the last layer of the model, without retraining. Leveraging concept whitening technique from Computer Vision combined with orthogonal projection, CALM removes unwanted latent directions associated with harmful content while preserving model performance. Experiments show that CALM reduces harmful outputs and outperforms baseline methods in most metrics, offering a lightweight approach to AI safety with no additional training data or model fine-tuning, while incurring only a small computational overhead at inference.

摘要: 大型语言模型容易受到绕过内置安全护栏的越狱攻击（例如，通过用对抗性提示欺骗模型）。我们提出了概念对齐和概念操纵CALM，这是一种推理时方法，通过修改模型最后一层的潜在表示来抑制有害概念，而无需重新训练。CALM利用计算机视觉的概念白化技术与垂直投影相结合，删除了与有害内容相关的不需要的潜在方向，同时保留了模型性能。实验表明，CALM减少了有害输出，并在大多数指标上优于基线方法，为人工智能安全提供了一种轻量级方法，无需额外的训练数据或模型微调，同时在推理时只产生很小的计算负担。



## **7. Autonomous Cyber Resilience via a Co-Evolutionary Arms Race within a Fortified Digital Twin Sandbox**

通过加强数字双沙盒内的协同进化军备竞赛增强自主网络韧性 cs.CR

6 pages, 2 figures, 4 equations, 1 algorithm, 3 tables, to be  published in ISPACS 2025, unabridged version exists as arXiv:2506.20102v1

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2506.20102v2) [paper-pdf](http://arxiv.org/pdf/2506.20102v2)

**Authors**: Malikussaid, Sutiyo

**Abstract**: The convergence of Information Technology and Operational Technology has exposed Industrial Control Systems to adaptive, intelligent adversaries that render static defenses obsolete. This paper introduces the Adversarial Resilience Co-evolution (ARC) framework, addressing the "Trinity of Trust" comprising model fidelity, data integrity, and analytical resilience. ARC establishes a co-evolutionary arms race within a Fortified Secure Digital Twin (F-SCDT), where a Deep Reinforcement Learning "Red Agent" autonomously discovers attack paths while an ensemble-based "Blue Agent" is continuously hardened against these threats. Experimental validation on the Tennessee Eastman Process (TEP) and Secure Water Treatment (SWaT) testbeds demonstrates superior performance in detecting novel attacks, with F1-scores improving from 0.65 to 0.89 and detection latency reduced from over 1200 seconds to 210 seconds. A comprehensive ablation study reveals that the co-evolutionary process itself contributes a 27% performance improvement. By integrating Explainable AI and proposing a Federated ARC architecture, this work presents a necessary paradigm shift toward dynamic, self-improving security for critical infrastructure.

摘要: 信息技术和运营技术的融合使工业控制系统暴露在自适应、智能的对手手中，从而使静态防御变得过时。本文介绍了对抗韧性协同进化（ARC）框架，解决了包括模型保真度、数据完整性和分析韧性在内的“信任三位一体”。ARC在强化安全数字孪生（F-SCDT）内建立了一场共同进化的军备竞赛，深度强化学习“红色代理”自主发现攻击路径，而基于集成的“蓝色代理”则持续加强针对这些威胁。田纳西州伊士曼工艺（TEP）和安全水处理（SWaT）测试台的实验验证表明，在检测新型攻击方面具有卓越的性能，F1评分从0.65提高到0.89，检测延迟从超过1200秒减少到210秒。一项全面的消融研究表明，共同进化过程本身可提高27%的性能。通过集成可解释人工智能并提出联邦ARC架构，这项工作为关键基础设施的动态、自我改进安全性提供了必要的范式转变。



## **8. Active Honeypot Guardrail System: Probing and Confirming Multi-Turn LLM Jailbreaks**

主动蜜罐保护系统：探测和识别多回合LLM越狱 cs.CR

6pages, 2 figures

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15017v1) [paper-pdf](http://arxiv.org/pdf/2510.15017v1)

**Authors**: ChenYu Wu, Yi Wang, Yang Liao

**Abstract**: Large language models (LLMs) are increasingly vulnerable to multi-turn jailbreak attacks, where adversaries iteratively elicit harmful behaviors that bypass single-turn safety filters. Existing defenses predominantly rely on passive rejection, which either fails against adaptive attackers or overly restricts benign users. We propose a honeypot-based proactive guardrail system that transforms risk avoidance into risk utilization. Our framework fine-tunes a bait model to generate ambiguous, non-actionable but semantically relevant responses, which serve as lures to probe user intent. Combined with the protected LLM's safe reply, the system inserts proactive bait questions that gradually expose malicious intent through multi-turn interactions. We further introduce the Honeypot Utility Score (HUS), measuring both the attractiveness and feasibility of bait responses, and use a Defense Efficacy Rate (DER) for balancing safety and usability. Initial experiment on MHJ Datasets with recent attack method across GPT-4o show that our system significantly disrupts jailbreak success while preserving benign user experience.

摘要: 大型语言模型（LLM）越来越容易受到多回合越狱攻击，对手会反复引发绕过单回合安全过滤器的有害行为。现有的防御主要依赖于被动拒绝，这要么无法抵御自适应攻击者，要么过度限制良性用户。我们提出了一种基于蜜罐的主动护栏系统，将风险规避转化为风险利用。我们的框架微调了诱饵模型，以生成模棱两可、不可操作但语义相关的响应，这些响应作为试探用户意图的诱饵。结合受保护的LLM的安全回复，系统插入主动诱饵问题，通过多回合交互逐渐暴露恶意意图。我们进一步引入了蜜罐效用评分（HUS），衡量诱饵反应的吸引力和可行性，并使用防御效能率（BER）来平衡安全性和可用性。针对GPT-4 o最近攻击方法的MTJ数据集的初步实验表明，我们的系统在保留良性用户体验的同时，显着破坏了越狱成功。



## **9. A Hard-Label Black-Box Evasion Attack against ML-based Malicious Traffic Detection Systems**

针对基于ML的恶意流量检测系统的硬标签黑匣子规避攻击 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14906v1) [paper-pdf](http://arxiv.org/pdf/2510.14906v1)

**Authors**: Zixuan Liu, Yi Zhao, Zhuotao Liu, Qi Li, Chuanpu Fu, Guangmeng Zhou, Ke Xu

**Abstract**: Machine Learning (ML)-based malicious traffic detection is a promising security paradigm. It outperforms rule-based traditional detection by identifying various advanced attacks. However, the robustness of these ML models is largely unexplored, thereby allowing attackers to craft adversarial traffic examples that evade detection. Existing evasion attacks typically rely on overly restrictive conditions (e.g., encrypted protocols, Tor, or specialized setups), or require detailed prior knowledge of the target (e.g., training data and model parameters), which is impractical in realistic black-box scenarios. The feasibility of a hard-label black-box evasion attack (i.e., applicable across diverse tasks and protocols without internal target insights) thus remains an open challenge. To this end, we develop NetMasquerade, which leverages reinforcement learning (RL) to manipulate attack flows to mimic benign traffic and evade detection. Specifically, we establish a tailored pre-trained model called Traffic-BERT, utilizing a network-specialized tokenizer and an attention mechanism to extract diverse benign traffic patterns. Subsequently, we integrate Traffic-BERT into the RL framework, allowing NetMasquerade to effectively manipulate malicious packet sequences based on benign traffic patterns with minimal modifications. Experimental results demonstrate that NetMasquerade enables both brute-force and stealthy attacks to evade 6 existing detection methods under 80 attack scenarios, achieving over 96.65% attack success rate. Notably, it can evade the methods that are either empirically or certifiably robust against existing evasion attacks. Finally, NetMasquerade achieves low-latency adversarial traffic generation, demonstrating its practicality in real-world scenarios.

摘要: 基于机器学习（ML）的恶意流量检测是一种有前途的安全范式。它通过识别各种高级攻击，优于基于规则的传统检测。然而，这些ML模型的稳健性在很大程度上尚未被探索，从而允许攻击者制作逃避检测的对抗性流量示例。现有的规避攻击通常依赖于过度限制的条件（例如，加密协议、Tor或专业设置），或者需要对目标的详细先验知识（例如，训练数据和模型参数），这在现实的黑匣子场景中是不切实际的。硬标签黑匣子规避攻击的可行性（即，适用于不同的任务和协议，而无需内部目标洞察）因此仍然是一个悬而未决的挑战。为此，我们开发了NetMasquerade，它利用强化学习（RL）来操纵攻击流以模拟良性流量并逃避检测。具体来说，我们建立了一个量身定制的预训练模型，称为“逻辑-BERT”，利用网络专用标记器和注意力机制来提取各种良性流量模式。随后，我们将Generic-BERT集成到RL框架中，允许NetMasquerade以最少的修改根据良性流量模式有效地操纵恶意数据包序列。实验结果表明，NetMasquerade能够在80种攻击场景下规避现有的6种检测方法，攻击成功率超过96.65%。值得注意的是，它可以规避针对现有规避攻击的经验性或可认证的方法。最后，NetMasquerade实现了低延迟对抗流量生成，展示了其在现实世界场景中的实用性。



## **10. Backdoor Unlearning by Linear Task Decomposition**

通过线性任务分解消除后门 cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14845v1) [paper-pdf](http://arxiv.org/pdf/2510.14845v1)

**Authors**: Amel Abdelraheem, Alessandro Favero, Gerome Bovet, Pascal Frossard

**Abstract**: Foundation models have revolutionized computer vision by enabling broad generalization across diverse tasks. Yet, they remain highly susceptible to adversarial perturbations and targeted backdoor attacks. Mitigating such vulnerabilities remains an open challenge, especially given that the large-scale nature of the models prohibits retraining to ensure safety. Existing backdoor removal approaches rely on costly fine-tuning to override the harmful behavior, and can often degrade performance on other unrelated tasks. This raises the question of whether backdoors can be removed without compromising the general capabilities of the models. In this work, we address this question and study how backdoors are encoded in the model weight space, finding that they are disentangled from other benign tasks. Specifically, this separation enables the isolation and erasure of the backdoor's influence on the model with minimal impact on clean performance. Building on this insight, we introduce a simple unlearning method that leverages such disentanglement. Through extensive experiments with CLIP-based models and common adversarial triggers, we show that, given the knowledge of the attack, our method achieves approximately perfect unlearning, while retaining, on average, 96% of clean accuracy. Additionally, we demonstrate that even when the attack and its presence are unknown, our method successfully unlearns backdoors by proper estimation using reverse-engineered triggers. Overall, our method consistently yields better unlearning and clean accuracy tradeoffs when compared to present state-of-the-art defenses.

摘要: 基础模型通过实现对不同任务的广泛概括，彻底改变了计算机视觉。然而，它们仍然极易受到对抗性扰动和有针对性的后门攻击的影响。缓解此类漏洞仍然是一个悬而未决的挑战，特别是考虑到模型的大规模性质禁止为确保安全而进行再培训。现有的后门删除方法依赖于昂贵的微调来覆盖有害行为，并且通常会降低其他不相关任务的性能。这引发了一个问题：是否可以在不损害模型的一般能力的情况下删除后门。在这项工作中，我们解决了这个问题，并研究后门如何在模型权重空间中编码，发现它们与其他良性任务脱钩。具体来说，这种分离可以隔离和消除后门对模型的影响，同时对清洁性能的影响最小。基于这一见解，我们引入了一种简单的去学习方法，可以利用这种解纠缠。通过对基于CLIP的模型和常见对抗触发器的广泛实验，我们表明，在了解攻击的情况下，我们的方法实现了近乎完美的去学习，同时平均保留了96%的准确性。此外，我们证明，即使攻击及其存在未知，我们的方法也可以通过使用反向工程触发器的正确估计来成功取消后门。总体而言，与目前最先进的防御相比，我们的方法始终能够产生更好的去学习和清晰的准确性权衡。



## **11. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

大型语言模型中对偏见激发的对抗鲁棒性进行基准测试：利用LLM作为评委的可扩展自动化评估 cs.CL

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2504.07887v2) [paper-pdf](http://arxiv.org/pdf/2504.07887v2)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: The growing integration of Large Language Models (LLMs) into critical societal domains has raised concerns about embedded biases that can perpetuate stereotypes and undermine fairness. Such biases may stem from historical inequalities in training data, linguistic imbalances, or adversarial manipulation. Despite mitigation efforts, recent studies show that LLMs remain vulnerable to adversarial attacks that elicit biased outputs. This work proposes a scalable benchmarking framework to assess LLM robustness to adversarial bias elicitation. Our methodology involves: (i) systematically probing models across multiple tasks targeting diverse sociocultural biases, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach, and (iii) employing jailbreak techniques to reveal safety vulnerabilities. To facilitate systematic benchmarking, we release a curated dataset of bias-related prompts, named CLEAR-Bias. Our analysis, identifying DeepSeek V3 as the most reliable judge LLM, reveals that bias resilience is uneven, with age, disability, and intersectional biases among the most prominent. Some small models outperform larger ones in safety, suggesting that training and architecture may matter more than scale. However, no model is fully robust to adversarial elicitation, with jailbreak attacks using low-resource languages or refusal suppression proving effective across model families. We also find that successive LLM generations exhibit slight safety gains, while models fine-tuned for the medical domain tend to be less safe than their general-purpose counterparts.

摘要: 大型语言模型（LLM）日益融入关键社会领域，引发了人们对嵌入式偏见的担忧，这些偏见可能会延续刻板印象并破坏公平性。此类偏见可能源于训练数据的历史不平等、语言不平衡或对抗操纵。尽管采取了缓解措施，但最近的研究表明，LLM仍然容易受到引发偏见输出的对抗攻击。这项工作提出了一个可扩展的基准框架来评估LLM对对抗性偏见引发的稳健性。我们的方法包括：（i）针对不同的社会文化偏见，系统地探索跨多项任务的模型，（ii）使用LLM作为法官的方法通过安全评分量化稳健性，以及（iii）采用越狱技术来揭示安全漏洞。为了促进系统性基准测试，我们发布了一个精心策划的偏差相关提示数据集，名为ClearAR-Bias。我们的分析将DeepSeek V3确定为最可靠的LLM法官，揭示了偏见复原力是不平衡的，其中年龄、残疾和交叉偏见是最突出的。一些小型模型在安全性方面优于大型模型，这表明培训和架构可能比规模更重要。然而，没有一个模型对对抗性诱导完全稳健，事实证明，使用低资源语言或拒绝抑制的越狱攻击在模型家族中都有效。我们还发现，连续几代LLM表现出轻微的安全性收益，而针对医疗领域微调的模型往往不如通用模型安全。



## **12. ATGen: Adversarial Reinforcement Learning for Test Case Generation**

ATGen：用于测试用例生成的对抗强化学习 cs.SE

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14635v1) [paper-pdf](http://arxiv.org/pdf/2510.14635v1)

**Authors**: Qingyao Li, Xinyi Dai, Weiwen Liu, Xiangyang Li, Yasheng Wang, Ruiming Tang, Yong Yu, Weinan Zhang

**Abstract**: Large Language Models (LLMs) excel at code generation, yet their outputs often contain subtle bugs, for which effective test cases are a critical bottleneck. Existing test generation methods, whether based on prompting or supervised fine-tuning, rely on static datasets. This imposes a ``fixed-difficulty ceiling'', fundamentally limiting their ability to uncover novel or more complex bugs beyond their training scope. To overcome this, we introduce ATGen, a framework that trains a test case generator via adversarial reinforcement learning. ATGen pits a test generator against an adversarial code generator that continuously crafts harder bugs to evade the current policy. This dynamic loop creates a curriculum of increasing difficulty challenging current policy. The test generator is optimized via Reinforcement Learning (RL) to jointly maximize ``Output Accuracy'' and ``Attack Success'', enabling it to learn a progressively stronger policy that breaks the fixed-difficulty ceiling of static training. Extensive experiments demonstrate that ATGen significantly outperforms state-of-the-art baselines. We further validate its practical utility, showing it serves as both a more effective filter for Best-of-N inference and a higher-quality reward source for training code generation models. Our work establishes a new, dynamic paradigm for improving the reliability of LLM-generated code.

摘要: 大型语言模型（LLM）擅长代码生成，但它们的输出通常包含微妙的bug，有效的测试用例是一个关键的瓶颈。现有的测试生成方法，无论是基于提示还是监督微调，都依赖于静态数据集。这强加了一个“固定难度上限”，从根本上限制了他们发现超出其培训范围的新的或更复杂的错误的能力。为了克服这一点，我们引入了ATGen，这是一个通过对抗性强化学习来训练测试用例生成器的框架。ATGen将测试生成器与对抗代码生成器进行了比较，后者不断地制造更难的错误来逃避当前政策。这种动态循环创造了挑战当前政策难度不断增加的课程。测试生成器通过强化学习（RL）进行优化，以共同最大化“输出准确度”和“攻击时间”，使其能够学习逐渐更强的策略，打破静态训练的固定难度上限。大量实验表明，ATGen的性能显着优于最先进的基线。我们进一步验证了它的实际实用性，表明它既是N最佳推理的更有效过滤器，也是训练代码生成模型的更高质量的奖励来源。我们的工作建立了一个新的动态范式，用于提高LLM生成的代码的可靠性。



## **13. SPIRIT: Patching Speech Language Models against Jailbreak Attacks**

精神：修补语音语言模型以防止越狱攻击 eess.AS

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2505.13541v2) [paper-pdf](http://arxiv.org/pdf/2505.13541v2)

**Authors**: Amirbek Djanibekov, Nurdaulet Mukhituly, Kentaro Inui, Hanan Aldarmaki, Nils Lukas

**Abstract**: Speech Language Models (SLMs) enable natural interactions via spoken instructions, which more effectively capture user intent by detecting nuances in speech. The richer speech signal introduces new security risks compared to text-based models, as adversaries can better bypass safety mechanisms by injecting imperceptible noise to speech. We analyze adversarial attacks and find that SLMs are substantially more vulnerable to jailbreak attacks, which can achieve a perfect 100% attack success rate in some instances. To improve security, we propose post-hoc patching defenses used to intervene during inference by modifying the SLM's activations that improve robustness up to 99% with (i) negligible impact on utility and (ii) without any re-training. We conduct ablation studies to maximize the efficacy of our defenses and improve the utility/security trade-off, validated with large-scale benchmarks unique to SLMs.

摘要: 语音语言模型（SLC）通过口头指令实现自然交互，通过检测语音中的细微差别更有效地捕捉用户意图。与基于文本的模型相比，更丰富的语音信号会带来新的安全风险，因为对手可以通过向语音注入难以感知的噪音来更好地绕过安全机制。我们分析了对抗性攻击，发现STM更容易受到越狱攻击，在某些情况下可以实现完美的100%攻击成功率。为了提高安全性，我们提出了事后修补防御，用于通过修改SPL的激活来在推理期间进行干预，从而将稳健性提高高达99%，并且（i）对效用的影响可以忽略不计，并且（ii）无需任何重新训练。我们进行消融研究，以最大限度地提高防御的功效并改善实用性/安全性权衡，并通过SLS特有的大规模基准进行验证。



## **14. Lost in the Averages: A New Specific Setup to Evaluate Membership Inference Attacks Against Machine Learning Models**

Lost in the April：一种新的特定设置，用于评估针对机器学习模型的成员推理攻击 cs.LG

Data Privacy Management workshop at ESORICS 2025

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2405.15423v2) [paper-pdf](http://arxiv.org/pdf/2405.15423v2)

**Authors**: Nataša Krčo, Florent Guépin, Matthieu Meeus, Bogdan Kulynych, Yves-Alexandre de Montjoye

**Abstract**: Synthetic data generators and machine learning models can memorize their training data, posing privacy concerns. Membership inference attacks (MIAs) are a standard method of estimating the privacy risk of these systems. The risk of individual records is typically computed by evaluating MIAs in a record-specific privacy game. We analyze the record-specific privacy game commonly used for evaluating attackers under realistic assumptions (the \textit{traditional} game) -- particularly for synthetic tabular data -- and show that it averages a record's privacy risk across datasets. We show this implicitly assumes the dataset a record is part of has no impact on the record's risk, providing a misleading risk estimate when a specific model or synthetic dataset is released. Instead, we propose a novel use of the leave-one-out game, used in existing work exclusively to audit differential privacy guarantees, and call this the \textit{model-seeded} game. We formalize it and show that it provides an accurate estimate of the privacy risk posed by a given adversary for a record in its specific dataset. We instantiate and evaluate the state-of-the-art MIA for synthetic data generators in the traditional and model-seeded privacy games, and show across multiple datasets and models that the two privacy games indeed result in different risk scores, with up to 94\% of high-risk records being overlooked by the traditional game. We further show that records in smaller datasets and models not protected by strong differential privacy guarantees tend to have a larger gap between risk estimates. Taken together, our results show that the model-seeded setup yields a risk estimate specific to a certain model or synthetic dataset released and in line with the standard notion of privacy leakage from prior work, meaningfully different from the dataset-averaged risk provided by the traditional privacy game.

摘要: 合成数据生成器和机器学习模型可以记住其训练数据，从而引发隐私问题。成员资格推断攻击（MIA）是估计这些系统隐私风险的标准方法。个人记录的风险通常是通过评估特定记录隐私游戏中的MIA来计算的。我们分析了通常用于在现实假设下评估攻击者的特定于记录的隐私游戏（\textit{traditional}游戏）--特别是对于合成表格数据--并表明它可以平均计算数据集中记录的隐私风险。我们表明，这隐含地假设记录所属的数据集对记录的风险没有影响，从而在发布特定模型或合成数据集时提供了误导性的风险估计。相反，我们提出了留一游戏的新颖用途，该游戏专门用于现有工作中来审计差异隐私保证，并称之为\textit{model-seed}游戏。我们将其形式化，并表明它可以准确估计特定对手对其特定数据集中的记录构成的隐私风险。我们实例化和评估了传统和模型种子隐私游戏中合成数据生成器的最新MIA，并在多个数据集和模型中表明，这两种隐私游戏确实会导致不同的风险评分，高达94%的高风险记录被传统游戏忽视。我们进一步表明，较小的数据集和模型中没有受到强差异隐私保证保护的记录往往在风险估计之间存在更大的差距。总而言之，我们的结果表明，模型种子设置会产生特定于发布的特定模型或合成数据集的风险估计，并且符合先前工作中隐私泄露的标准概念，与传统隐私游戏提供的厕所平均风险有意义地不同。



## **15. Certifying optimal MEV strategies with Lean**

通过精益认证最佳MEV策略 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14480v1) [paper-pdf](http://arxiv.org/pdf/2510.14480v1)

**Authors**: Massimo Bartoletti, Riccardo Marchesin, Roberto Zunino

**Abstract**: Maximal Extractable Value (MEV) refers to a class of attacks to decentralized applications where the adversary profits by manipulating the ordering, inclusion, or exclusion of transactions in a blockchain. Decentralized Finance (DeFi) protocols are a primary target of these attacks, as their logic depends critically on transaction sequencing. To date, MEV attacks have already extracted billions of dollars in value, underscoring their systemic impact on blockchain security. Verifying the absence of MEV attacks requires determining suitable upper bounds, i.e. proving that no adversarial strategy can extract more value (if any) than expected by protocol designers. This problem is notoriously difficult: the space of adversarial strategies is extremely vast, making empirical studies and pen-and-paper reasoning insufficiently rigorous. In this paper, we present the first mechanized formalization of MEV in the Lean theorem prover. We introduce a methodology to construct machine-checked proofs of MEV bounds, providing correctness guarantees beyond what is possible with existing techniques. To demonstrate the generality of our approach, we model and analyse the MEV of two paradigmatic DeFi protocols. Notably, we develop the first machine-checked proof of the optimality of sandwich attacks in Automated Market Makers, a fundamental DeFi primitive.

摘要: 最大可提取值（MEV）是指对去中心化应用程序的一类攻击，其中对手通过操纵区块链中交易的排序、包含或排除来获利。去中心化金融（DeFi）协议是这些攻击的主要目标，因为它们的逻辑严重依赖于交易排序。迄今为止，MEV攻击已经损失了数十亿美元的价值，凸显了它们对区块链安全的系统性影响。确认不存在MEV攻击需要确定合适的上限，即证明没有对抗策略可以提取比协议设计者预期更多的价值（如果有的话）。这个问题是出了名的困难：对抗策略的空间极其广阔，使得实证研究和纸笔推理不够严格。本文中，我们在精益定理证明器中首次提出了MEV的机械化形式化。我们引入了一种方法来构建MEV界限的机器检查证明，提供超出现有技术可能范围的正确性保证。为了证明我们方法的通用性，我们对两种范式DeFi协议的MEV进行了建模和分析。值得注意的是，我们在Automated Market Makers（一个基本的DeFi基元）中开发了第一个三明治攻击最优性的机器检查证明。



## **16. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

没有对抗防御的对抗防御：通过实例级主成分去除增强语言模型稳健性 cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2507.21750v4) [paper-pdf](http://arxiv.org/pdf/2507.21750v4)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.

摘要: 预训练的语言模型（PLM）推动了自然语言处理的重大进展，但仍然容易受到对抗攻击，引发了对其在现实世界应用程序中稳健性的担忧。之前的研究试图通过隐式或显式地在训练过程中引入对抗性扰动来减轻对抗性攻击的影响。虽然这两种策略都增强了稳健性，但它们通常会产生很高的计算成本。在这项工作中，我们提出了一个简单而有效的附加模块，该模块通过删除实例级主成分来增强PLM的对抗鲁棒性，而不依赖于传统的对抗防御或干扰原始训练数据。我们的方法将嵌入空间转换为逼近高斯属性，从而降低其对对抗性扰动的敏感性，同时保留语义关系。这种转换以一种最小化对抗性噪音对决策边界的影响的方式对齐嵌入分布，增强稳健性，而无需对抗性示例或昂贵的训练时间扩展。对八个基准数据集的评估表明，我们的方法提高了对抗稳健性，同时保持了与基线相当的攻击前准确性，实现了稳健性和概括性之间的平衡。



## **17. The Fluorescent Veil: A Stealthy and Effective Physical Adversarial Patch Against Traffic Sign Recognition**

荧光面纱：针对交通标志识别的隐形有效物理对抗补丁 cs.CV

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2409.12394v2) [paper-pdf](http://arxiv.org/pdf/2409.12394v2)

**Authors**: Shuai Yuan, Xingshuo Han, Hongwei Li, Guowen Xu, Wenbo Jiang, Tao Ni, Qingchuan Zhao, Yuguang Fang

**Abstract**: Recently, traffic sign recognition (TSR) systems have become a prominent target for physical adversarial attacks. These attacks typically rely on conspicuous stickers and projections, or using invisible light and acoustic signals that can be easily blocked. In this paper, we introduce a novel attack medium, i.e., fluorescent ink, to design a stealthy and effective physical adversarial patch, namely FIPatch, to advance the state-of-the-art. Specifically, we first model the fluorescence effect in the digital domain to identify the optimal attack settings, which guide the real-world fluorescence parameters. By applying a carefully designed fluorescence perturbation to the target sign, the attacker can later trigger a fluorescent effect using invisible ultraviolet light, causing the TSR system to misclassify the sign and potentially leading to traffic accidents. We conducted a comprehensive evaluation to investigate the effectiveness of FIPatch, which shows a success rate of 98.31% in low-light conditions. Furthermore, our attack successfully bypasses five popular defenses and achieves a success rate of 96.72%.

摘要: 最近，交通标志识别（TSB）系统已成为物理对抗攻击的主要目标。这些攻击通常依赖于显眼的贴纸和投影，或者使用不可见光和可以轻易阻止的声信号。在本文中，我们引入了一种新型的攻击媒介，即荧光墨水，设计一种隐形有效的物理对抗补丁，即FIPatch，以推进最新技术水平。具体来说，我们首先在数字领域对荧光效应进行建模，以识别最佳攻击设置，从而指导现实世界的荧光参数。通过对目标标志应用精心设计的荧光扰动，攻击者随后可以使用不可见的紫外光触发荧光效应，导致TSB系统错误分类该标志，并可能导致交通事故。我们进行了全面的评估来调查FIPache的有效性，结果显示在弱光条件下的成功率为98.31%。此外，我们的攻击成功绕过了五种流行防御，成功率达到96.72%。



## **18. Structured Universal Adversarial Attacks on Object Detection for Video Sequences**

视频序列对象检测的结构化通用对抗攻击 cs.CV

Accepted at GCPR 2025 (German Conference on Pattern Recognition).  This is a different version as submitted to the conference, not the official  conference proceedings

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14460v1) [paper-pdf](http://arxiv.org/pdf/2510.14460v1)

**Authors**: Sven Jacob, Weijia Shao, Gjergji Kasneci

**Abstract**: Video-based object detection plays a vital role in safety-critical applications. While deep learning-based object detectors have achieved impressive performance, they remain vulnerable to adversarial attacks, particularly those involving universal perturbations. In this work, we propose a minimally distorted universal adversarial attack tailored for video object detection, which leverages nuclear norm regularization to promote structured perturbations concentrated in the background. To optimize this formulation efficiently, we employ an adaptive, optimistic exponentiated gradient method that enhances both scalability and convergence. Our results demonstrate that the proposed attack outperforms both low-rank projected gradient descent and Frank-Wolfe based attacks in effectiveness while maintaining high stealthiness. All code and data are publicly available at https://github.com/jsve96/AO-Exp-Attack.

摘要: 基于视频的对象检测在安全关键应用中发挥着至关重要的作用。虽然基于深度学习的对象检测器取得了令人印象深刻的性能，但它们仍然容易受到对抗攻击，特别是涉及普适扰动的攻击。在这项工作中，我们提出了一种为视频对象检测量身定制的最低失真的通用对抗攻击，它利用核规范正规化来促进集中在背景中的结构化扰动。为了有效地优化该公式，我们采用了自适应、乐观的取指数梯度方法，以增强可扩展性和收敛性。我们的结果表明，所提出的攻击在有效性上优于低等级投影梯度下降和基于Frank-Wolfe的攻击，同时保持了高的隐蔽性。所有代码和数据均可在https://github.com/jsve96/AO-Exp-Attack上公开获取。



## **19. A Multi-domain Image Translative Diffusion StyleGAN for Iris Presentation Attack Detection**

用于虹膜呈现攻击检测的多域图像翻译扩散风格GAN cs.CV

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14314v1) [paper-pdf](http://arxiv.org/pdf/2510.14314v1)

**Authors**: Shivangi Yadav, Arun Ross

**Abstract**: An iris biometric system can be compromised by presentation attacks (PAs) where artifacts such as artificial eyes, printed eye images, or cosmetic contact lenses are presented to the system. To counteract this, several presentation attack detection (PAD) methods have been developed. However, there is a scarcity of datasets for training and evaluating iris PAD techniques due to the implicit difficulties in constructing and imaging PAs. To address this, we introduce the Multi-domain Image Translative Diffusion StyleGAN (MID-StyleGAN), a new framework for generating synthetic ocular images that captures the PA and bonafide characteristics in multiple domains such as bonafide, printed eyes and cosmetic contact lens. MID-StyleGAN combines the strengths of diffusion models and generative adversarial networks (GANs) to produce realistic and diverse synthetic data. Our approach utilizes a multi-domain architecture that enables the translation between bonafide ocular images and different PA domains. The model employs an adaptive loss function tailored for ocular data to maintain domain consistency. Extensive experiments demonstrate that MID-StyleGAN outperforms existing methods in generating high-quality synthetic ocular images. The generated data was used to significantly enhance the performance of PAD systems, providing a scalable solution to the data scarcity problem in iris and ocular biometrics. For example, on the LivDet2020 dataset, the true detect rate at 1% false detect rate improved from 93.41% to 98.72%, showcasing the impact of the proposed method.

摘要: 虹膜生物识别系统可能会受到呈现攻击（PA）的损害，其中人造眼睛、印刷眼睛图像或美容隐形眼镜等伪影被呈现给系统。为了对抗这一点，人们开发了多种呈现攻击检测（PAD）方法。然而，由于PA的构建和成像存在隐性困难，用于训练和评估虹膜PED技术的数据集稀缺。为了解决这个问题，我们引入了多域图像转换扩散StyleGAN（MID-StyleGAN），这是一个用于生成合成眼部图像的新框架，可以捕捉多个领域中的PA和Bonafide特征，例如Bonafide、印刷眼睛和美容隐形眼镜。MID-StyleGAN结合了扩散模型和生成式对抗网络（GAN）的优势，生成真实且多样化的合成数据。我们的方法利用多域架构，能够在真实的眼部图像和不同PA域之间进行翻译。该模型采用为眼部数据量身定制的自适应损失函数来保持域一致性。大量实验表明，MID-StyleGAN在生成高质量合成眼部图像方面优于现有方法。生成的数据用于显着增强PAD系统的性能，为虹膜和眼部生物识别中的数据稀缺问题提供可扩展的解决方案。例如，在LivDet 2020数据集上，1%错误检测率时的真检测率从93.41%提高到98.72%，展示了所提出方法的影响。



## **20. Impact of Regularization on Calibration and Robustness: from the Representation Space Perspective**

正规化对校准和鲁棒性的影响：从表示空间的角度来看 cs.CV

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2410.03999v2) [paper-pdf](http://arxiv.org/pdf/2410.03999v2)

**Authors**: Jonghyun Park, Juyeop Kim, Jong-Seok Lee

**Abstract**: Recent studies have shown that regularization techniques using soft labels, e.g., label smoothing, Mixup, and CutMix, not only enhance image classification accuracy but also mitigate miscalibration due to overconfident predictions, and improve robustness against adversarial attacks. However, the underlying mechanisms of such improvements remain underexplored. In this paper, we offer a novel explanation from the perspective of the representation space (i.e., the space of the features obtained at the penultimate layer). Based on examination of decision boundaries and structure of features (or representation vectors), our study investigates confidence contours and gradient directions within the representation space. Furthermore, we analyze the adjustments in feature distributions due to regularization in relation to these contours and directions, from which we uncover central mechanisms inducing improved calibration and robustness. Our findings provide new insights into the characteristics of the high-dimensional representation space in relation to training and regularization using soft labels.

摘要: 最近的研究表明，使用软标签的正规化技术，例如，标签平滑、Mixup和CutMix不仅提高了图像分类准确性，还减轻了由于过于自信的预测造成的误校准，并提高了对抗攻击的鲁棒性。然而，此类改进的根本机制仍然没有得到充分的探索。在本文中，我们从表示空间的角度提供了一种新颖的解释（即，在倒数第二层获得的特征的空间）。基于对决策边界和特征（或表示载体）结构的检查，我们的研究调查了表示空间内的置信度轮廓和梯度方向。此外，我们分析了由于与这些轮廓和方向相关的规则化而对特征分布的调整，从中我们发现了导致校准和鲁棒性改进的核心机制。我们的研究结果为与使用软标签的训练和正规化相关的多维表示空间的特征提供了新的见解。



## **21. RHINO: Guided Reasoning for Mapping Network Logs to Adversarial Tactics and Techniques with Large Language Models**

RHINO：将网络条件映射到具有大型语言模型的对抗策略和技术的引导推理 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14233v1) [paper-pdf](http://arxiv.org/pdf/2510.14233v1)

**Authors**: Fanchao Meng, Jiaping Gui, Yunbo Li, Yue Wu

**Abstract**: Modern Network Intrusion Detection Systems generate vast volumes of low-level alerts, yet these outputs remain semantically fragmented, requiring labor-intensive manual correlation with high-level adversarial behaviors. Existing solutions for automating this mapping-rule-based systems and machine learning classifiers-suffer from critical limitations: rule-based approaches fail to adapt to novel attack variations, while machine learning methods lack contextual awareness and treat tactic-technique mapping as a syntactic matching problem rather than a reasoning task. Although Large Language Models have shown promise in cybersecurity tasks, preliminary experiments reveal that existing LLM-based methods frequently hallucinate technique names or produce decontextualized mappings due to their single-step classification approach.   To address these challenges, we introduce RHINO, a novel framework that decomposes LLM-based attack analysis into three interpretable phases mirroring human reasoning: (1) behavioral abstraction, where raw logs are translated into contextualized narratives; (2) multi-role collaborative inference, generating candidate techniques by evaluating behavioral evidence against MITRE ATT&CK knowledge; and (3) validation, cross-referencing predictions with official MITRE definitions to rectify hallucinations. RHINO bridges the semantic gap between low-level observations and adversarial intent while improving output reliability through structured reasoning.   We evaluate RHINO on three benchmarks across four backbone models. RHINO achieved high accuracy, with model performance ranging from 86.38% to 88.45%, resulting in relative gains from 24.25% to 76.50% across different models. Our results demonstrate that RHINO significantly enhances the interpretability and scalability of threat analysis, offering a blueprint for deploying LLMs in operational security settings.

摘要: 现代网络入侵检测系统会生成大量低级警报，但这些输出仍然是语义碎片化的，需要与高级对抗行为进行劳动密集型的手动关联。用于自动化这种映射的现有解决方案--基于规则的系统和机器学习分类器--存在严重的局限性：基于规则的方法无法适应新颖的攻击变体，而机器学习方法缺乏上下文感知，并将战术技术映射视为语法匹配问题而不是推理任务。尽管大型语言模型在网络安全任务中表现出了希望，但初步实验表明，现有的基于LLM的方法由于其分步分类方法而经常使技术名称产生幻觉或产生去上下文化映射。   为了应对这些挑战，我们引入了RHINO，这是一个新颖的框架，它将基于LLM的攻击分析分解为反映人类推理的三个可解释阶段：（1）行为抽象，其中原始日志被翻译为上下文化叙述;（2）多角色协作推理，通过针对MITRE ATT & CK知识评估行为证据来生成候选技术;以及（3）验证，将预测与官方MITRE定义交叉引用以纠正幻觉。RHINO弥合了低级观察和对抗意图之间的语义差距，同时通过结构化推理提高了输出的可靠性。   我们对四种主干模型的三个基准进行了评估。RHINO实现了高准确度，模型性能范围为86.38%至88.45%，不同型号的相对收益范围为24.25%至76.50%。我们的结果表明，RHINO显着增强了威胁分析的可解释性和可扩展性，为在运营安全环境中部署LLM提供了蓝图。



## **22. SoK: Adversarial Evasion Attacks Practicality in NIDS Domain and the Impact of Dynamic Learning**

SoK：NIDS领域的对抗性规避攻击的实用性以及动态学习的影响 cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2306.05494v4) [paper-pdf](http://arxiv.org/pdf/2306.05494v4)

**Authors**: Mohamed elShehaby, Ashraf Matrawy

**Abstract**: Machine Learning (ML) has become pervasive, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy compared to traditional models in processing and classifying large volumes of data. However, ML has been found to have several flaws, most importantly, adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the suitability of these attacks against ML-based network security entities, especially NIDS, due to the wide difference between different domains regarding the generation of adversarial attacks.   To further explore the practicality of adversarial attacks against ML-based NIDS in-depth, this paper presents several key contributions: identifying numerous practicality issues for evasion adversarial attacks on ML-NIDS using an attack tree threat model, introducing a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS, identifying specific leaf nodes in our attack tree that demonstrate some practicality for real-world implementation and conducting a comprehensive review and exploration of these potentially viable attack approaches, and investigating how the dynamicity of real-world ML models affects evasion adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effectiveness of adversarial attacks. While adversarial attacks can compromise ML-based NIDSs, our aim is to highlight the significant gap between research and real-world practicality in this domain, which warrants attention.

摘要: 机器学习（ML）已变得普遍，与处理和分类大量数据的传统模型相比，由于其自动化性质和高准确性，其在网络入侵检测系统（NIDS）中的部署是不可避免的。然而，ML被发现存在几个缺陷，最重要的是对抗性攻击，旨在欺骗ML模型产生错误的预测。虽然大多数对抗性攻击研究都集中在计算机视觉数据集，但由于不同领域之间关于对抗性攻击的生成存在巨大差异，最近的研究探索了这些攻击针对基于ML的网络安全实体（尤其是NIDS）的适用性。   为了进一步深入探讨针对基于ML的NIDS的对抗攻击的实用性，本文提出了几个关键贡献：使用攻击树威胁模型识别规避对ML-NIDS的对抗攻击的众多实用性问题，引入与针对基于ML的NIDS的对抗攻击相关的实用性问题分类法，识别我们的攻击树中的特定叶节点，这些节点展示了现实世界实施的一些实用性，并对这些潜在可行的攻击方法进行全面审查和探索，并研究现实世界ML模型的动态性如何影响规避针对NIDS的对抗攻击。我们的实验表明，即使没有对抗训练，持续的重新训练也会降低对抗攻击的有效性。虽然对抗性攻击可能会损害基于ML的NIDS，但我们的目标是强调该领域的研究与现实世界实用性之间的显着差距，这值得关注。



## **23. Proof-Carrying Fair Ordering: Asymmetric Verification for BFT via Incremental Graphs**

携带证明公平排序：通过增量图进行BFT的非对称验证 cs.DC

18 pages, 4 figures

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14186v1) [paper-pdf](http://arxiv.org/pdf/2510.14186v1)

**Authors**: Pengkun Ren, Hai Dong, Nasrin Sohrabi, Zahir Tari, Pengcheng Zhang

**Abstract**: Byzantine Fault-Tolerant (BFT) consensus protocols ensure agreement on transaction ordering despite malicious actors, but unconstrained ordering power enables sophisticated value extraction attacks like front running and sandwich attacks - a critical threat to blockchain systems. Order-fair consensus curbs adversarial value extraction by constraining how leaders may order transactions. While state-of-the-art protocols such as Themis attain strong guarantees through graph-based ordering, they ask every replica to re-run the leader's expensive ordering computation for validation - an inherently symmetric and redundant paradigm. We present AUTIG, a high-performance, pluggable order-fairness service that breaks this symmetry. Our key insight is that verifying a fair order does not require re-computing it. Instead, verification can be reduced to a stateless audit of succinct, verifiable assertions about the ordering graph's properties. AUTIG realizes this via an asymmetric architecture: the leader maintains a persistent Unconfirmed-Transaction Incremental Graph (UTIG) to amortize graph construction across rounds and emits a structured proof of fairness with each proposal; followers validate the proof without maintaining historical state. AUTIG introduces three critical innovations: (i) incremental graph maintenance driven by threshold-crossing events and state changes; (ii) a decoupled pipeline that overlaps leader-side collection/update/extraction with follower-side stateless verification; and (iii) a proof design covering all internal pairs in the finalized prefix plus a frontier completeness check to rule out hidden external dependencies. We implement AUTIG and evaluate it against symmetric graph-based baselines under partial synchrony. Experiments show higher throughput and lower end-to-end latency while preserving gamma-batch-order-fairness.

摘要: 拜占庭的故障容忍（BFT）共识协议确保在交易排序上达成一致，尽管存在恶意行为者，但不受约束的排序能力允许复杂的价值提取攻击，例如前置运行和三明治攻击--这是区块链系统的严重威胁。订单公平共识通过限制领导者下令交易的方式来限制对抗性价值的提取。虽然Themis等最先进的协议通过基于图的排序获得了强有力的保证，但它们要求每个副本重新运行领导者昂贵的排序计算以进行验证--这是一种本质上对称且冗余的范式。我们介绍了AUTIG，这是一种高性能、可插入的订单公平服务，打破了这种对称性。我们的主要见解是，验证公平顺序并不需要重新计算它。相反，验证可以简化为对有关排序图属性的简洁、可验证断言的无状态审计。AUTIG通过非对称架构实现了这一点：领导者维护一个持久的未确认交易增量图（UTIG），以在各轮之间摊销图构造，并针对每个提案发出结构化的公平性证明;追随者在不维护历史状态的情况下验证证明。AUTIG引入了三项关键创新：（i）由阈值跨越事件和状态变化驱动的增量图维护;（ii）一个脱钩的管道，将领导者端收集/更新/提取与后续端无状态验证重叠;（iii）覆盖最终前置中所有内部对的证明设计加上边界完整性检查以排除隐藏的外部依赖关系。我们实现AUTIG并在部分同步下根据基于对称图形的基线对其进行评估。实验表明，更高的吞吐量和更低的端到端延迟，同时保持伽玛批次顺序公平性。



## **24. Laser Fault Injection in Memristor-Based Accelerators for AI/ML and Neuromorphic Computing**

用于AI/ML和神经形态计算的记忆存储器加速器中的激光故障注入 cs.ET

3 pages, 4 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.14120v1) [paper-pdf](http://arxiv.org/pdf/2510.14120v1)

**Authors**: Muhammad Faheemur Rahman, Wayne Burleson

**Abstract**: Memristive crossbar arrays (MCA) are emerging as efficient building blocks for in-memory computing and neuromorphic hardware due to their high density and parallel analog matrix-vector multiplication capabilities. However, the physical properties of their nonvolatile memory elements introduce new attack surfaces, particularly under fault injection scenarios. This work explores Laser Fault Injection as a means of inducing analog perturbations in MCA-based architectures. We present a detailed threat model in which adversaries target memristive cells to subtly alter their physical properties or outputs using laser beams. Through HSPICE simulations of a large MCA on 45 nm CMOS tech. node, we show how laser-induced photocurrent manifests in output current distributions, enabling differential fault analysis to infer internal weights with up to 99.7% accuracy, replicate the model, and compromise computational integrity through targeted weight alterations by approximately 143%.

摘要: 记忆交叉杆阵列（MCA）因其高密度和并行模拟矩阵-载体相乘能力而成为内存计算和神经形态硬件的有效构建模块。然而，其非易失性存储器元件的物理属性引入了新的攻击表面，特别是在故障注入场景下。这项工作探索了激光故障注入作为在基于MCA的架构中引发模拟扰动的一种方法。我们提出了一个详细的威胁模型，其中对手瞄准记忆细胞，使用激光束微妙地改变它们的物理性质或输出。通过HSPICE模拟基于45纳米互补金属氧化物半导体技术的大型MCA。节点，我们展示了激光感生光电流如何在输出电流分布中体现，使差异故障分析能够以高达99.7%的准确度推断内部权重，复制模型，并通过目标权重改变约143%来损害计算完整性。



## **25. Resource-Aware Stealthy Attacks in Vehicle Platoons**

车辆排中的资源感知隐形攻击 eess.SY

13 pages, 8 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.14119v1) [paper-pdf](http://arxiv.org/pdf/2510.14119v1)

**Authors**: Ali Eslami, Mohammad Pirani

**Abstract**: Connected and Autonomous Vehicles (CAVs) are transforming modern transportation by enabling cooperative applications such as vehicle platooning, where multiple vehicles travel in close formation to improve efficiency and safety. However, the heavy reliance on inter-vehicle communication makes platoons highly susceptible to attacks, where even subtle manipulations can escalate into severe physical consequences. While existing research has largely focused on defending against attacks, far less attention has been given to stealthy adversaries that aim to covertly manipulate platoon behavior. This paper introduces a new perspective on the attack design problem by demonstrating how attackers can guide platoons toward their own desired trajectories while remaining undetected. We outline conditions under which such attacks are feasible, analyze their dependence on communication topologies and control protocols, and investigate the resources required by the attacker. By characterizing the resources needed to launch stealthy attacks, we address system vulnerabilities and informing the design of resilient countermeasures. Our findings reveal critical weaknesses in current platoon architectures and anomaly detection mechanisms and provide methods to develop more secure and trustworthy CAV systems.

摘要: 互联和自动驾驶汽车（Cavs）正在通过实现车辆排队等协作应用来改变现代交通，其中多辆车辆紧密排列行驶，以提高效率和安全性。然而，对车内通信的严重依赖使得排极易受到攻击，即使是微妙的操纵也可能升级为严重的身体后果。虽然现有的研究主要集中在防御攻击上，但对旨在秘密操纵排行为的隐形对手的关注要少得多。本文通过演示攻击者如何引导排走向自己想要的轨迹，同时保持不被发现，引入了攻击设计问题的新视角。我们概述的条件下，这种攻击是可行的，分析其依赖于通信拓扑结构和控制协议，并调查所需的攻击者的资源。通过描述发动隐形攻击所需的资源，我们解决了系统漏洞，并为弹性对策的设计提供了信息。我们的研究结果揭示了当前排体系结构和异常检测机制的关键弱点，并提供了开发更安全，更值得信赖的CAV系统的方法。



## **26. A Survey and Future Outlook on Indoor Location Fingerprinting Privacy Preservation**

室内位置指纹隐私保护的调查与未来展望 cs.CR

Published in Computer Networks

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2404.07345v2) [paper-pdf](http://arxiv.org/pdf/2404.07345v2)

**Authors**: Amir Fathalizadeh, Vahideh Moghtadaiee, Mina Alishahi

**Abstract**: The pervasive integration of Indoor Positioning Systems (IPS) arises from the limitations of Global Navigation Satellite Systems (GNSS) in indoor environments, leading to the widespread adoption of Location-Based Services (LBS) in places such as shopping malls, airports, hospitals, museums, corporate campuses, and smart buildings. Specifically, indoor location fingerprinting (ILF) systems employ diverse signal fingerprints from user devices, enabling precise location identification by Location Service Providers (LSP). Despite its broad applications across various domains, ILF introduces a notable privacy risk, as both LSP and potential adversaries inherently have access to this sensitive information, compromising users' privacy. Consequently, concerns regarding privacy vulnerabilities in this context necessitate a focused exploration of privacy-preserving mechanisms. In response to these concerns, this survey presents a comprehensive review of Indoor Location Fingerprinting Privacy-Preserving Mechanisms (ILFPPM) based on cryptographic, anonymization, differential privacy (DP), and federated learning (FL) techniques. We also propose a distinctive and novel grouping of privacy vulnerabilities, adversary models, privacy attacks, and evaluation metrics specific to ILF systems. Given the identified limitations and research gaps in this survey, we highlight numerous prospective opportunities for future investigation, aiming to motivate researchers interested in advancing ILF systems. This survey constitutes a valuable reference for researchers and provides a clear overview for those beyond this specific research domain. To further help the researchers, we have created an online resource repository, which can be found at \href{https://github.com/amir-ftlz/ilfppm}{https://github.com/amir-ftlz/ilfppm}.

摘要: 室内定位系统（IPS）的普遍集成源于全球导航卫星系统（GNSS）在室内环境中的局限性，导致位置服务（RBS）在购物中心、机场、医院、博物馆、企业园区和智能建筑等场所广泛采用。具体来说，室内位置指纹识别（ILF）系统采用来自用户设备的不同信号指纹，从而实现位置服务提供商（Laser）的精确位置识别。尽管ILF在各个领域有广泛的应用，但它带来了显着的隐私风险，因为STP和潜在对手本质上都可以访问这些敏感信息，从而损害了用户的隐私。因此，在这种情况下对隐私漏洞的担忧需要重点探索隐私保护机制。为了回应这些担忧，这项调查对基于加密、匿名化、差异隐私（DP）和联邦学习（FL）技术的室内位置指纹隐私保护机制（ILFPPM）进行了全面审查。我们还提出了一种独特且新颖的隐私漏洞分组、对手模型、隐私攻击和特定于ILF系统的评估指标。鉴于本调查中发现的局限性和研究差距，我们强调了未来研究的众多潜在机会，旨在激励对推进ILF系统感兴趣的研究人员。这项调查为研究人员提供了宝贵的参考，并为这一特定研究领域以外的人提供了清晰的概述。为了进一步帮助研究人员，我们创建了一个在线资源库，可以在\href{https：//github.com/amir-ftlz/ilfppm}{https：//github.com/amir-ftlz/ilfppm}上找到。



## **27. Signature in Code Backdoor Detection, how far are we?**

代码后门检测签名，我们还走多远？ cs.SE

20 pages, 3 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13992v1) [paper-pdf](http://arxiv.org/pdf/2510.13992v1)

**Authors**: Quoc Hung Le, Thanh Le-Cong, Bach Le, Bowen Xu

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into software development workflows, they also become prime targets for adversarial attacks. Among these, backdoor attacks are a significant threat, allowing attackers to manipulate model outputs through hidden triggers embedded in training data. Detecting such backdoors remains a challenge, and one promising approach is the use of Spectral Signature defense methods that identify poisoned data by analyzing feature representations through eigenvectors. While some prior works have explored Spectral Signatures for backdoor detection in neural networks, recent studies suggest that these methods may not be optimally effective for code models. In this paper, we revisit the applicability of Spectral Signature-based defenses in the context of backdoor attacks on code models. We systematically evaluate their effectiveness under various attack scenarios and defense configurations, analyzing their strengths and limitations. We found that the widely used setting of Spectral Signature in code backdoor detection is often suboptimal. Hence, we explored the impact of different settings of the key factors. We discovered a new proxy metric that can more accurately estimate the actual performance of Spectral Signature without model retraining after the defense.

摘要: 随着大型语言模型（LLM）越来越多地集成到软件开发工作流程中，它们也成为对抗性攻击的主要目标。其中，后门攻击是一个重大威胁，允许攻击者通过嵌入训练数据中的隐藏触发器操纵模型输出。检测此类后门仍然是一个挑战，一种有前途的方法是使用光谱签名防御方法，该方法通过特征载体分析特征表示来识别有毒数据。虽然之前的一些工作已经探索了用于神经网络后门检测的光谱签名，但最近的研究表明，这些方法对于代码模型可能不是最佳有效的。在本文中，我们重新审视了基于光谱签名的防御在代码模型后门攻击的背景下的适用性。我们系统地评估它们在各种攻击场景和防御配置下的有效性，分析它们的优势和局限性。我们发现，代码后门检测中广泛使用的光谱签名设置通常是次优的。因此，我们探讨了不同设置对关键因素的影响。我们发现了一种新的代理指标，它可以更准确地估计Spectral Signature的实际性能，而无需在防御后进行模型重新训练。



## **28. All Code, No Thought: Current Language Models Struggle to Reason in Ciphered Language**

全代码，不思考：当前语言模型在Cipbitt语言中难以推理 cs.CL

Version 2: updated related works section on LLM steganography

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.09714v2) [paper-pdf](http://arxiv.org/pdf/2510.09714v2)

**Authors**: Shiyuan Guo, Henry Sleight, Fabien Roger

**Abstract**: Detecting harmful AI actions is important as AI agents gain adoption. Chain-of-thought (CoT) monitoring is one method widely used to detect adversarial attacks and AI misalignment. However, attackers and misaligned models might evade CoT monitoring through ciphered reasoning: reasoning hidden in encrypted, translated, or compressed text. To assess this risk, we test whether models can perform ciphered reasoning. For each of 28 different ciphers, we fine-tune and prompt up to 10 models to reason in that cipher. We measure model accuracy on math problems as a proxy for reasoning ability. Across the models we test, we find an asymmetry: model accuracy can drop significantly when reasoning in ciphered text, even though models demonstrate comprehension of ciphered text by being able to translate it accurately to English. Even frontier models struggle with lesser-known ciphers, although they can reason accurately in well-known ciphers like rot13. We show that ciphered reasoning capability correlates with cipher prevalence in pretraining data. We also identify scaling laws showing that ciphered reasoning capability improves slowly with additional fine-tuning data. Our work suggests that evading CoT monitoring using ciphered reasoning may be an ineffective tactic for current models and offers guidance on constraining the development of this capability in future frontier models.

摘要: 随着人工智能代理的采用，检测有害的人工智能行为非常重要。思想链（CoT）监控是广泛用于检测对抗攻击和人工智能失调的一种方法。然而，攻击者和错位的模型可能会通过加密推理来逃避CoT监控：推理隐藏在加密、翻译或压缩文本中。为了评估这种风险，我们测试模型是否可以执行加密推理。对于28个不同的密码中的每一个，我们都会微调并提示多达10个模型对该密码进行推理。我们衡量数学问题的模型准确性，作为推理能力的代表。在我们测试的模型中，我们发现了一种不对称性：在加密文本中进行推理时，模型准确性可能会显着下降，尽管模型通过能够将其准确地翻译成英语来证明对加密文本的理解。即使是前沿模型也难以应对鲜为人知的密码，尽管它们可以在像rot13这样的知名密码中准确推理。我们表明，加密推理能力与预训练数据中的密码流行率相关。我们还确定了缩放定律，表明加密推理能力随着额外的微调数据而缓慢提高。我们的工作表明，使用加密推理逃避CoT监控对于当前模型来说可能是一种无效的策略，并为限制未来前沿模型中这种能力的发展提供了指导。



## **29. Provably Invincible Adversarial Attacks on Reinforcement Learning Systems: A Rate-Distortion Information-Theoretic Approach**

对强化学习系统的可证明无敌的对抗攻击：一种速率失真信息理论方法 cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13792v1) [paper-pdf](http://arxiv.org/pdf/2510.13792v1)

**Authors**: Ziqing Lu, Lifeng Lai, Weiyu Xu

**Abstract**: Reinforcement learning (RL) for the Markov Decision Process (MDP) has emerged in many security-related applications, such as autonomous driving, financial decisions, and drone/robot algorithms. In order to improve the robustness/defense of RL systems against adversaries, studying various adversarial attacks on RL systems is very important. Most previous work considered deterministic adversarial attack strategies in MDP, which the recipient (victim) agent can defeat by reversing the deterministic attacks. In this paper, we propose a provably ``invincible'' or ``uncounterable'' type of adversarial attack on RL. The attackers apply a rate-distortion information-theoretic approach to randomly change agents' observations of the transition kernel (or other properties) so that the agent gains zero or very limited information about the ground-truth kernel (or other properties) during the training. We derive an information-theoretic lower bound on the recipient agent's reward regret and show the impact of rate-distortion attacks on state-of-the-art model-based and model-free algorithms. We also extend this notion of an information-theoretic approach to other types of adversarial attack, such as state observation attacks.

摘要: 马尔科夫决策过程（MDP）的强化学习（RL）已出现在许多与安全相关的应用中，例如自动驾驶、金融决策和无人机/机器人算法。为了提高RL系统对对手的鲁棒性/防御性，研究对RL系统的各种对抗攻击非常重要。之前的大多数工作都考虑了MDP中的确定性对抗攻击策略，接收者（受害者）代理可以通过逆转确定性攻击来击败这些策略。在本文中，我们提出了一种可证明的对RL的“无敌”或“不可对抗”类型的对抗性攻击。攻击者应用率失真信息论方法来随机改变代理对过渡核（或其他属性）的观察，以便代理在训练期间获得零或非常有限的有关地面真值核（或其他属性）的信息。我们推导出接收代理奖励后悔的信息理论下限，并展示了速率失真攻击对最先进的基于模型和无模型算法的影响。我们还将信息论方法的概念扩展到其他类型的对抗性攻击，例如状态观察攻击。



## **30. Towards Adversarial Robustness and Uncertainty Quantification in DINOv2-based Few-Shot Anomaly Detection**

基于DINOv2的少镜头异常检测中的对抗鲁棒性和不确定性量化 cs.CV

10 pages, 5 figures, 3 tables

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13643v1) [paper-pdf](http://arxiv.org/pdf/2510.13643v1)

**Authors**: Akib Mohammed Khan, Bartosz Krawczyk

**Abstract**: Foundation models such as DINOv2 have shown strong performance in few-shot anomaly detection, yet two key questions remain unexamined: (i) how susceptible are these detectors to adversarial perturbations; and (ii) how well do their anomaly scores reflect calibrated uncertainty? Building on AnomalyDINO, a training-free deep nearest-neighbor detector over DINOv2 features, we present one of the first systematic studies of adversarial attacks and uncertainty estimation in this setting. To enable white-box gradient attacks while preserving test-time behavior, we attach a lightweight linear head to frozen DINOv2 features only for crafting perturbations. Using this heuristic, we evaluate the impact of FGSM across the MVTec-AD and VisA datasets and observe consistent drops in F1, AUROC, AP, and G-mean, indicating that imperceptible perturbations can flip nearest-neighbor relations in feature space to induce confident misclassification. Complementing robustness, we probe reliability and find that raw anomaly scores are poorly calibrated, revealing a gap between confidence and correctness that limits safety-critical use. As a simple, strong baseline toward trustworthiness, we apply post-hoc Platt scaling to the anomaly scores for uncertainty estimation. The resulting calibrated posteriors yield significantly higher predictive entropy on adversarially perturbed inputs than on clean ones, enabling a practical flagging mechanism for attack detection while reducing calibration error (ECE). Our findings surface concrete vulnerabilities in DINOv2-based few-shot anomaly detectors and establish an evaluation protocol and baseline for robust, uncertainty-aware anomaly detection. We argue that adversarial robustness and principled uncertainty quantification are not optional add-ons but essential capabilities if anomaly detection systems are to be trustworthy and ready for real-world deployment.

摘要: DINOv 2等基础模型在少量异常检测中表现出了强劲的性能，但有两个关键问题仍未得到研究：（i）这些检测器对对抗性扰动的敏感性有多大;（ii）它们的异常分数反映了校准的不确定性？基于AnomalyDINO（一种针对DINOv 2特征的免训练深度近邻检测器），我们在这种环境下首次对对抗攻击和不确定性估计进行了系统性研究之一。为了启用白盒梯度攻击，同时保留测试时行为，我们将轻量级线性头附加到冻结的DINOv 2特征上，仅用于制造扰动。使用这种启发式，我们评估了FGSM对MMVTec-AD和VisA数据集的影响，并观察到F1、AUROC、AP和G-mean的一致下降，这表明不可感知的扰动可以翻转特征空间中的最近邻关系，从而引发可信的错误分类。除了鲁棒性之外，我们还调查了可靠性，发现原始异常分数校准不良，揭示了信心和正确性之间的差距，从而限制了安全关键使用。作为可信度的简单、强基线，我们将事后普拉特缩放应用于异常分数以进行不确定性估计。由此产生的校准后验在对抗干扰的输入上产生的预测信息比在干净的输入上明显更高，从而实现了实用的攻击检测标记机制，同时减少了校准误差（ECA）。我们的研究结果揭示了基于DINOv 2的几次异常检测器中的具体漏洞，并为稳健的、不确定性感知的异常检测建立了评估协议和基线。我们认为，如果异常检测系统要值得信赖并为现实世界的部署做好准备，对抗稳健性和原则性的不确定性量化不是可选的附加组件，而是必不可少的能力。



## **31. Selective Adversarial Attacks on LLM Benchmarks**

LLM基准的选择性对抗攻击 cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13570v1) [paper-pdf](http://arxiv.org/pdf/2510.13570v1)

**Authors**: Ivan Dubrovsky, Anastasia Orlova, Illarion Iov, Nina Gubina, Irena Gureeva, Alexey Zaytsev

**Abstract**: Benchmarking outcomes increasingly govern trust, selection, and deployment of LLMs, yet these evaluations remain vulnerable to semantically equivalent adversarial perturbations. Prior work on adversarial robustness in NLP has emphasized text attacks that affect many models equally, leaving open the question of whether it is possible to selectively degrade or enhance performance while minimally affecting other models. We formalize this problem and study selective adversarial attacks on MMLU - a widely used benchmark designed to measure a language model's broad general knowledge and reasoning ability across different subjects. Using canonical attacks integrated into TextAttack framework, we introduce a protocol for selectivity assessment, develop a custom constraint to increase selectivity of attacks and propose a surrogate-LLM pipeline that generates selective perturbations. Empirically, we find that selective adversarial attacks exist and can materially alter relative rankings, challenging the fairness, reproducibility, and transparency of leaderboard-driven evaluation. Our results motivate perturbation-aware reporting and robustness diagnostics for LLM evaluation and demonstrate that even subtle edits can shift comparative judgments.

摘要: 基准结果越来越多地影响着LLM的信任、选择和部署，但这些评估仍然容易受到语义等效的对抗性扰动的影响。之前关于NLP对抗鲁棒性的工作强调了对许多模型同等影响的文本攻击，这留下了一个问题：是否可以选择性地降低或增强性能，同时对其他模型的影响最小。我们将这个问题形式化，并研究对MMLU的选择性对抗攻击，MMLU是一个广泛使用的基准，旨在衡量语言模型在不同学科中广泛的常识和推理能力。使用集成到文本攻击框架中的规范攻击，我们引入了选择性评估协议，开发自定义约束以增加攻击的选择性，并提出了生成选择性扰动的代理LLM管道。从经验上讲，我们发现存在选择性对抗性攻击，并且可以实质性地改变相对排名，挑战排行榜驱动的评估的公平性，可重复性和透明度。我们的研究结果激发了LLM评估的扰动感知报告和鲁棒性诊断，并表明即使是细微的编辑也可以改变比较判断。



## **32. Systematic Literature Review on Vehicular Collaborative Perception - A Computer Vision Perspective**

车辆协作感知的系统文献综述--计算机视觉视角 cs.CV

38 pages, 8 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2504.04631v2) [paper-pdf](http://arxiv.org/pdf/2504.04631v2)

**Authors**: Lei Wan, Jianxin Zhao, Andreas Wiedholz, Manuel Bied, Mateus Martinez de Lucena, Abhishek Dinkar Jagtap, Andreas Festag, Antônio Augusto Fröhlich, Hannan Ejaz Keen, Alexey Vinel

**Abstract**: The effectiveness of autonomous vehicles relies on reliable perception capabilities. Despite significant advancements in artificial intelligence and sensor fusion technologies, current single-vehicle perception systems continue to encounter limitations, notably visual occlusions and limited long-range detection capabilities. Collaborative Perception (CP), enabled by Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication, has emerged as a promising solution to mitigate these issues and enhance the reliability of autonomous systems. Beyond advancements in communication, the computer vision community is increasingly focusing on improving vehicular perception through collaborative approaches. However, a systematic literature review that thoroughly examines existing work and reduces subjective bias is still lacking. Such a systematic approach helps identify research gaps, recognize common trends across studies, and inform future research directions. In response, this study follows the PRISMA 2020 guidelines and includes 106 peer-reviewed articles. These publications are analyzed based on modalities, collaboration schemes, and key perception tasks. Through a comparative analysis, this review illustrates how different methods address practical issues such as pose errors, temporal latency, communication constraints, domain shifts, heterogeneity, and adversarial attacks. Furthermore, it critically examines evaluation methodologies, highlighting a misalignment between current metrics and CP's fundamental objectives. By delving into all relevant topics in-depth, this review offers valuable insights into challenges, opportunities, and risks, serving as a reference for advancing research in vehicular collaborative perception.

摘要: 自动驾驶汽车的有效性依赖于可靠的感知能力。尽管人工智能和传感器融合技术取得了重大进步，但当前的单车感知系统继续遇到局限性，特别是视觉遮挡和有限的远程检测能力。由车对车（V2 V）和车对基础设施（V2 I）通信实现的协作感知（CP）已成为缓解这些问题并提高自主系统可靠性的一种有前途的解决方案。除了通信领域的进步之外，计算机视觉界越来越关注通过协作方法改善车辆感知。然而，仍然缺乏彻底审查现有工作并减少主观偏见的系统性文献审查。这种系统性方法有助于识别研究差距、识别研究中的共同趋势，并为未来的研究方向提供信息。作为回应，这项研究遵循PRISMA 2020指南，包括106篇同行评审的文章。这些出版物是根据模式、协作方案和关键感知任务进行分析的。通过比较分析，本综述说明了不同的方法如何解决实际问题，例如姿势错误、时间延迟、通信约束、域转移、异类和对抗性攻击。此外，它还批判性地审查了评估方法，强调了当前指标与CP基本目标之间的不一致。通过深入研究所有相关主题，本评论对挑战、机遇和风险提供了宝贵的见解，为推进车辆协作感知研究提供参考。



## **33. Towards Quantum Enhanced Adversarial Robustness with Rydberg Reservoir Learnin**

利用Rydberg水库学习实现量子增强对抗鲁棒性 quant-ph

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13473v1) [paper-pdf](http://arxiv.org/pdf/2510.13473v1)

**Authors**: Shehbaz Tariq, Muhammad Talha, Symeon Chatzinotas, Hyundong Shin

**Abstract**: Quantum reservoir computing (QRC) leverages the high-dimensional, nonlinear dynamics inherent in quantum many-body systems for extracting spatiotemporal patterns in sequential and time-series data with minimal training overhead. Although QRC inherits the expressive capabilities associated with quantum encodings, recent studies indicate that quantum classifiers based on variational circuits remain susceptible to adversarial perturbations. In this perspective, we investigate the first systematic evaluation of adversarial robustness in a QRC based learning model. Our reservoir comprises an array of strongly interacting Rydberg atoms governed by a fixed Hamiltonian, which naturally evolves under complex quantum dynamics, producing high-dimensional embeddings. A lightweight multilayer perceptron serves as the trainable readout layer. We utilize the balanced datasets, namely MNIST, Fashion-MNIST, and Kuzushiji-MNIST, as a benchmark for rigorously evaluating the impact of augmenting the quantum reservoir with a Multilayer perceptron (MLP) in white-box adversarial attacks to assess its robustness. We demonstrate that this approach yields significantly higher accuracy than purely classical models across all perturbation strengths tested. This hybrid approach reveals a new source of quantum advantage and

摘要: 量子储层计算（QRC）利用量子多体系统固有的多维、非线性动力学，以最小的训练负担提取顺序和时间序列数据中的时空模式。尽管QRC继承了与量子编码相关的表达能力，但最近的研究表明，基于变分电路的量子分类器仍然容易受到对抗性扰动的影响。从这个角度来看，我们研究了基于QRC的学习模型中对对抗稳健性的首次系统评估。我们的水库由一系列强相互作用的里德堡原子组成，这些原子由固定的汉密尔顿量控制，该Hamilton量在复杂的量子动力学下自然进化，产生多维嵌入。轻量级的多层感知器充当可训练的读出层。我们利用平衡数据集（即MNIST、Fashion-MNIST和Kuzushiji-MNIST）作为基准，严格评估在白盒对抗攻击中使用多层感知器（MLP）增强量子库的影响，以评估其稳健性。我们证明，在所有测试的扰动强度下，这种方法比纯经典模型产生了显着更高的准确性。这种混合方法揭示了量子优势的新来源，



## **34. Generalist++: A Meta-learning Framework for Mitigating Trade-off in Adversarial Training**

通才++：一个用于缓解对抗性培训中权衡的元学习框架 cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13361v1) [paper-pdf](http://arxiv.org/pdf/2510.13361v1)

**Authors**: Yisen Wang, Yichuan Mo, Hongjun Wang, Junyi Li, Zhouchen Lin

**Abstract**: Despite the rapid progress of neural networks, they remain highly vulnerable to adversarial examples, for which adversarial training (AT) is currently the most effective defense. While AT has been extensively studied, its practical applications expose two major limitations: natural accuracy tends to degrade significantly compared with standard training, and robustness does not transfer well across attacks crafted under different norm constraints. Unlike prior works that attempt to address only one issue within a single network, we propose to partition the overall generalization goal into multiple sub-tasks, each assigned to a dedicated base learner. By specializing in its designated objective, each base learner quickly becomes an expert in its field. In the later stages of training, we interpolate their parameters to form a knowledgeable global learner, while periodically redistributing the global parameters back to the base learners to prevent their optimization trajectories from drifting too far from the shared target. We term this framework Generalist and introduce three variants tailored to different application scenarios. Both theoretical analysis and extensive experiments demonstrate that Generalist achieves lower generalization error and significantly alleviates the trade-off problems compared with baseline methods. Our results suggest that Generalist provides a promising step toward developing fully robust classifiers in the future.

摘要: 尽管神经网络进步迅速，但它们仍然极易受到对抗性例子的影响，而对抗性训练（AT）是目前最有效的防御。虽然AT已经得到了广泛的研究，但其实际应用暴露了两个主要局限性：与标准训练相比，自然准确性往往会显着下降，并且鲁棒性在不同规范约束下设计的攻击中无法很好地转移。与之前试图在单个网络中仅解决一个问题的作品不同，我们建议将总体概括目标划分为多个子任务，每个子任务分配给一个专用的基本学习器。通过专注于其指定的目标，每个基础学习者都会很快成为其领域的专家。在训练的后期阶段，我们对它们的参数进行插值，以形成知识渊博的全局学习器，同时定期将全局参数重新分配回基本学习器，以防止它们的优化轨迹偏离共享目标太远。我们将此框架命名为Generalist，并引入了针对不同应用场景量身定制的三个变体。理论分析和大量实验都表明，与基线方法相比，Generalist实现了更低的概括误差，并显着减轻了权衡问题。我们的结果表明，Generalist为未来开发完全稳健的分类器迈出了有希望的一步。



## **35. SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models**

SafeGuider：针对文本到图像模型的稳健且实用的内容安全控制 cs.CR

Accepted by ACM CCS 2025, Code is available at [this https  URL](https://github.com/pgqihere/safeguider)

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.05173v3) [paper-pdf](http://arxiv.org/pdf/2510.05173v3)

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce SafeGuider, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, SafeGuider generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.

摘要: 文本到图像模型在从自然语言描述生成高质量图像方面表现出了非凡的能力。然而，这些模型非常容易受到对抗提示的影响，这可能会绕过安全措施并产生有害内容。尽管有各种防御策略，但在现实世界应用程序中保持实用性的同时实现针对攻击的鲁棒性仍然是一个重大挑战。为了解决这个问题，我们首先对稳定扩散（SD）模型中的文本编码器进行了实证研究，该模型是一种广泛使用且具有代表性的文本到图像模型。我们的研究结果表明，[EOS]令牌充当语义聚合器，在其嵌入空间中的良性提示和对抗提示之间表现出明显的分布模式。基于这一见解，我们引入了SafeGuider，这是一个两步框架，旨在在不影响发电质量的情况下进行稳健的安全控制。SafeGuider将嵌入级识别模型与安全意识特征擦除束搜索算法相结合。此集成使该框架能够为良性提示维持高质量图像生成，同时确保针对域内和域外攻击的强大防御。SafeGuider在最大限度地降低攻击成功率方面表现出出色的有效性，在各种攻击场景中实现的最高攻击成功率仅为5.48%。此外，SafeGuider不会拒绝为不安全提示生成或产生黑色图像，而是生成安全且有意义的图像，增强了其实际实用性。此外，SafeGuider不限于SD模型，可以有效应用于其他文本到图像模型，例如Flux模型，展示了其在不同架构中的通用性和适应性。我们希望SafeGuider能够为安全文本到图像系统的实际部署提供一些线索。



## **36. SAJA: A State-Action Joint Attack Framework on Multi-Agent Deep Reinforcement Learning**

SAJA：一个基于多智能体深度强化学习的状态-动作联合攻击框架 cs.AI

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13262v1) [paper-pdf](http://arxiv.org/pdf/2510.13262v1)

**Authors**: Weiqi Guo, Guanjun Liu, Ziyuan Zhou

**Abstract**: Multi-Agent Deep Reinforcement Learning (MADRL) has shown potential for cooperative and competitive tasks such as autonomous driving and strategic gaming. However, models trained by MADRL are vulnerable to adversarial perturbations on states and actions. Therefore, it is essential to investigate the robustness of MADRL models from an attack perspective. Existing studies focus on either state-only attacks or action-only attacks, but do not consider how to effectively joint them. Simply combining state and action perturbations such as randomly perturbing states and actions does not exploit their potential synergistic effects. In this paper, we propose the State-Action Joint Attack (SAJA) framework that has a good synergistic effects. SAJA consists of two important phases: (1) In the state attack phase, a multi-step gradient ascent method utilizes both the actor network and the critic network to compute an adversarial state, and (2) in the action attack phase, based on the perturbed state, a second gradient ascent uses the critic network to craft the final adversarial action. Additionally, a heuristic regularizer measuring the distance between the perturbed actions and the original clean ones is added into the loss function to enhance the effectiveness of the critic's guidance. We evaluate SAJA in the Multi-Agent Particle Environment (MPE), demonstrating that (1) it outperforms and is more stealthy than state-only or action-only attacks, and (2) existing state or action defense methods cannot defend its attacks.

摘要: 多智能体深度强化学习（MADRL）已显示出自动驾驶和战略游戏等合作和竞争任务的潜力。然而，MADRL训练的模型很容易受到状态和动作的对抗性扰动的影响。因此，从攻击的角度研究MADRL模型的稳健性至关重要。现有的研究要么关注仅针对状态的攻击，要么关注仅针对动作的攻击，但没有考虑如何有效地将它们联合起来。简单地组合状态和动作扰动（例如随机扰动状态和动作）并不能利用其潜在的协同效应。本文提出了具有良好协同效应的状态行动联合攻击（SAJA）框架。SAJA由两个重要阶段组成：（1）在状态攻击阶段，多步梯度上升方法利用行动者网络和评论者网络来计算对抗状态，（2）在动作攻击阶段，基于受干扰的状态，第二次梯度上升使用评论者网络来制作最终的对抗动作。此外，在损失函数中添加了一个启发式正规化器，用于测量受干扰的动作与原始干净的动作之间的距离，以增强评论家指导的有效性。我们在多智能体粒子环境（MBE）中评估了SAJA，证明（1）它优于仅状态或仅动作攻击，并且比仅状态攻击更隐蔽，并且（2）现有的状态或动作防御方法无法防御其攻击。



## **37. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

个人可以操纵多主体的集体决策吗？ cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2509.16494v2) [paper-pdf](http://arxiv.org/pdf/2509.16494v2)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.

摘要: 个体大型语言模型（LLM）已在医疗保健和法律等各个领域展现出强大的能力。最近的研究还表明，协调的多智能体系统通过协作表现出增强的决策和推理能力。然而，由于单个LLM的脆弱性以及访问多代理系统中所有代理的困难，出现了一个关键问题：如果攻击者只知道一个代理，他们还能生成能够误导集体决策的对抗样本吗？为了探索这个问题，我们将其描述为一个信息不完整的游戏，其中攻击者只知道一个目标代理，并且缺乏对系统中其他代理的了解。通过这个公式，我们提出了M-Spoiler，这是一个模拟多智能体系统内的智能体交互以生成对抗样本的框架。然后使用这些样本来操纵目标系统中的目标代理，误导系统的协作决策过程。更具体地说，M-Spoiler引入了一种顽固代理，它通过模拟目标系统中代理的潜在顽固反应来积极帮助优化对抗样本。这增强了生成的对抗样本误导系统的有效性。通过针对各种任务的广泛实验，我们的研究结果证实了多代理系统中单个代理的知识所带来的风险，并证明了我们框架的有效性。我们还探索了几种防御机制，表明我们提出的攻击框架仍然比基线更有效，强调了进一步研究防御策略的必要性。



## **38. Model-agnostic Adversarial Attack and Defense for Vision-Language-Action Models**

视觉-语言-动作模型的模型不可知的对抗攻击和防御 cs.CV

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13237v1) [paper-pdf](http://arxiv.org/pdf/2510.13237v1)

**Authors**: Haochuan Xu, Yun Sing Koh, Shuhuai Huang, Zirun Zhou, Di Wang, Jun Sakuma, Jingfeng Zhang

**Abstract**: Vision-Language-Action (VLA) models have achieved revolutionary progress in robot learning, enabling robots to execute complex physical robot tasks from natural language instructions. Despite this progress, their adversarial robustness remains underexplored. In this work, we propose both adversarial patch attack and corresponding defense strategies for VLA models. We first introduce the Embedding Disruption Patch Attack (EDPA), a model-agnostic adversarial attack that generates patches directly placeable within the camera's view. In comparison to prior methods, EDPA can be readily applied to different VLA models without requiring prior knowledge of the model architecture, or the controlled robotic manipulator. EDPA constructs these patches by (i) disrupting the semantic alignment between visual and textual latent representations, and (ii) maximizing the discrepancy of latent representations between adversarial and corresponding clean visual inputs. Through the optimization of these objectives, EDPA distorts the VLA's interpretation of visual information, causing the model to repeatedly generate incorrect actions and ultimately result in failure to complete the given robotic task. To counter this, we propose an adversarial fine-tuning scheme for the visual encoder, in which the encoder is optimized to produce similar latent representations for both clean and adversarially perturbed visual inputs. Extensive evaluations on the widely recognized LIBERO robotic simulation benchmark demonstrate that EDPA substantially increases the task failure rate of cutting-edge VLA models, while our proposed defense effectively mitigates this degradation. The codebase is accessible via the homepage at https://edpa-attack.github.io/.

摘要: 视觉-语言-动作（VLA）模型在机器人学习方面取得了革命性的进展，使机器人能够从自然语言指令执行复杂的物理机器人任务。尽管取得了这一进展，但它们的对抗鲁棒性仍然没有得到充分的研究。在这项工作中，我们提出了对抗补丁攻击和相应的防御策略VLA模型。我们首先介绍嵌入中断补丁攻击（EDPA），这是一种与模型无关的对抗性攻击，可以生成可直接放置在相机视图中的补丁。与现有方法相比，EDPA可以很容易地应用于不同的VLA模型，而不需要模型架构或受控机器人操纵器的先验知识。EDPA通过（i）破坏视觉和文本潜在表示之间的语义对齐，以及（ii）最大化对抗性和相应的干净视觉输入之间潜在表示的差异来构建这些补丁。通过优化这些目标，EDPA扭曲了VLA对视觉信息的解释，导致模型反复生成错误的动作，最终导致无法完成给定的机器人任务。为了解决这个问题，我们提出了一种针对视觉编码器的对抗性微调方案，其中编码器经过优化，以为干净和对抗干扰的视觉输入产生类似的潜在表示。对广泛认可的LIBERO机器人仿真基准的广泛评估表明，EDPA大大增加了尖端VLA模型的任务失败率，而我们提出的防御有效地减轻了这种退化。代码库可通过主页https://edpa-attack.github.io/访问。



## **39. SHIELD: Classifier-Guided Prompting for Robust and Safer LVLMs**

SHIELD：分类器引导的预算，实现更强大、更安全的LVLM cs.CL

Preprint

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13190v1) [paper-pdf](http://arxiv.org/pdf/2510.13190v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Large Vision-Language Models (LVLMs) unlock powerful multimodal reasoning but also expand the attack surface, particularly through adversarial inputs that conceal harmful goals in benign prompts. We propose SHIELD, a lightweight, model-agnostic preprocessing framework that couples fine-grained safety classification with category-specific guidance and explicit actions (Block, Reframe, Forward). Unlike binary moderators, SHIELD composes tailored safety prompts that enforce nuanced refusals or safe redirection without retraining. Across five benchmarks and five representative LVLMs, SHIELD consistently lowers jailbreak and non-following rates while preserving utility. Our method is plug-and-play, incurs negligible overhead, and is easily extendable to new attack types -- serving as a practical safety patch for both weakly and strongly aligned LVLMs.

摘要: 大型视觉语言模型（LVLM）解锁了强大的多模式推理，但也扩大了攻击面，特别是通过在良性提示中隐藏有害目标的对抗性输入。我们提出SHIELD，这是一个轻量级的、模型不可知的预处理框架，它将细粒度的安全分类与特定类别的指导和显式动作（Block、Reframe、Forward）结合起来。与二元版主不同，SHIELD编写了量身定制的安全提示，无需再培训即可强制执行细致入微的拒绝或安全重定向。在五个基准和五个有代表性的LVLM中，SHIELD持续降低越狱和不跟随率，同时保持实用性。我们的方法是即插即用的，所产生的负担可以忽略不计，并且可以轻松扩展到新的攻击类型--作为弱对齐和强对齐LVLM的实用安全补丁。



## **40. Improving Transferability of Adversarial Examples via Bayesian Attacks**

通过Bayesian攻击提高对抗性示例的可移植性 cs.LG

Accepted by TCSVT

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2307.11334v2) [paper-pdf](http://arxiv.org/pdf/2307.11334v2)

**Authors**: Qizhang Li, Yiwen Guo, Xiaochen Yang, Wangmeng Zuo, Hao Chen

**Abstract**: The transferability of adversarial examples allows for the attack on unknown deep neural networks (DNNs), posing a serious threat to many applications and attracting great attention. In this paper, we improve the transferability of adversarial examples by incorporating the Bayesian formulation into both the model parameters and model input, enabling their joint diversification. We demonstrate that combination of Bayesian formulations for both the model input and model parameters yields significant improvements in transferability. By introducing advanced approximations of the posterior distribution over the model input, adversarial transferability achieves further enhancement, surpassing all state-of-the-arts when attacking without model fine-tuning. Additionally, we propose a principled approach to fine-tune model parameters within this Bayesian framework. Extensive experiments demonstrate that our method achieves a new state-of-the-art in transfer-based attacks, significantly improving the average success rate on ImageNet and CIFAR-10. Code at: https://github.com/qizhangli/MoreBayesian-jrnl.

摘要: 对抗性示例的可移植性允许对未知深度神经网络（DNN）进行攻击，对许多应用构成严重威胁并引起了极大关注。在本文中，我们通过将Bayesian公式融入模型参数和模型输入中来提高对抗性示例的可移植性，从而实现它们的联合多样化。我们证明，模型输入和模型参数的Bayesian公式的组合可以显着提高可移植性。通过在模型输入上引入后验分布的高级逼近，对抗可转移性实现了进一步的增强，在无需模型微调的情况下进行攻击时超越了所有最新技术水平。此外，我们还提出了一种有原则的方法来在此Bayesian框架内微调模型参数。大量实验表明，我们的方法在基于传输的攻击方面实现了新的最新水平，显着提高了ImageNet和CIFAR-10上的平均成功率。代码：https://github.com/qizhangli/MoreBayesian-jrnl。



## **41. Privacy-Aware Framework of Robust Malware Detection in Indoor Robots: Hybrid Quantum Computing and Deep Neural Networks**

室内机器人鲁棒恶意软件检测的隐私感知框架：混合量子计算和深度神经网络 cs.CR

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13136v1) [paper-pdf](http://arxiv.org/pdf/2510.13136v1)

**Authors**: Tan Le, Van Le, Sachin Shetty

**Abstract**: Indoor robotic systems within Cyber-Physical Systems (CPS) are increasingly exposed to Denial of Service (DoS) attacks that compromise localization, control and telemetry integrity. We propose a privacy-aware malware detection framework for indoor robotic systems, which leverages hybrid quantum computing and deep neural networks to counter DoS threats in CPS, while preserving privacy information. By integrating quantum-enhanced feature encoding with dropout-optimized deep learning, our architecture achieves up to 95.2% detection accuracy under privacy-constrained conditions. The system operates without handcrafted thresholds or persistent beacon data, enabling scalable deployment in adversarial environments. Benchmarking reveals robust generalization, interpretability and resilience against training instability through modular circuit design. This work advances trustworthy AI for secure, autonomous CPS operations.

摘要: 网络物理系统（CPS）内的室内机器人系统越来越容易受到拒绝服务（DPS）攻击，从而损害定位、控制和遥感完整性。我们为室内机器人系统提出了一种隐私感知恶意软件检测框架，该框架利用混合量子计算和深度神经网络来对抗CPS中的DPS威胁，同时保护隐私信息。通过将量子增强特征编码与辍学优化深度学习集成，我们的架构在隐私受限的条件下实现了高达95.2%的检测准确率。该系统无需手工制作的阈值或持久信标数据即可运行，从而能够在对抗环境中进行可扩展部署。基准测试通过模块化电路设计揭示了鲁棒的概括性、可解释性和针对训练不稳定性的弹性。这项工作推动了值得信赖的人工智能，以实现安全、自主的CPS操作。



## **42. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments**

RedTeamCUA：混合Web-OS环境中计算机使用代理的真实对抗测试 cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2505.21936v3) [paper-pdf](http://arxiv.org/pdf/2505.21936v3)

**Authors**: Zeyi Liao, Jaylen Jones, Linxi Jiang, Yuting Ning, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun

**Abstract**: Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning high ASRs in realistic end-to-end settings, with the strongest-to-date Claude 4.5 Sonnet | CUA exhibiting the highest ASR of 60%, indicating that CUA threats can already result in tangible risks to users and computer systems. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.

摘要: 计算机使用代理（CUA）承诺在操作系统（OS）和网络上自动执行复杂任务，但仍然容易受到间接提示注入的影响。当前对该威胁的评估要么缺乏对现实但受控的环境的支持，要么忽视了涉及两个接口的混合Web操作系统攻击场景。为了解决这个问题，我们提出了RedTeamCUA，这是一个对抗性测试框架，具有新颖的混合沙盒，该沙盒将基于虚拟机的操作系统环境与基于Docker的Web平台集成在一起。我们的沙箱支持为红色团队定制的关键功能，例如灵活的对抗场景配置，以及通过在对抗注入时直接初始化测试来将对抗评估与CUA的导航限制分开的设置。使用RedTeamCUA，我们开发RTC-Bench，这是一个包含864个示例的综合基准测试，可以调查现实的混合Web操作系统攻击场景和基本安全漏洞。对当前前沿CUA进行基准测试发现重大漏洞：Claude 3.7十四行诗|CUA的ASB为42.9%，而受评估的最安全的CUA Operator仍为7.6%。值得注意的是，CUA经常尝试执行尝试率高达92.5%的对抗任务，尽管由于能力限制而未能完成这些任务。尽管如此，我们在现实的端到端环境中观察到了很高的ASB，其中包括迄今为止最强的Claude 4.5十四行诗|CUA表现出最高的ASB为60%，这表明CUA威胁已经可能给用户和计算机系统带来切实的风险。总体而言，RedTeamCUA为推进对CUA漏洞的现实、受控和系统性分析提供了一个重要框架，强调了在现实世界部署之前对间接提示注入进行强有力的防御的迫切需要。



## **43. Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT)**

文本对抗示例的可解释性和透明性驱动的检测和转换（IT-DT） cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2307.01225v2) [paper-pdf](http://arxiv.org/pdf/2307.01225v2)

**Authors**: Bushra Sabir, M. Ali Babar, Sharif Abuadbba

**Abstract**: Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into non-adversarial counterparts that align with the model's intended behavior while preserving the text's meaning. Transparency is emphasized through human expert involvement. Experts review and provide feedback on detection and transformation results, enhancing decision-making, especially in complex scenarios. The framework generates insights and threat intelligence empowering analysts to identify vulnerabilities and improve model robustness. Comprehensive experiments demonstrate the effectiveness of IT-DT in detecting and transforming adversarial examples. The approach enhances interpretability, provides transparency, and enables accurate identification and successful transformation of adversarial inputs. By combining technical analysis and human expertise, IT-DT significantly improves the resilience and trustworthiness of transformer-based text classifiers against adversarial attacks.

摘要: BERT、Roberta、T5和GPT-3等基于转换器的文本分类器在NLP中表现出令人印象深刻的性能。然而，它们对对抗示例的脆弱性构成了安全风险。现有的防御方法缺乏可解释性，因此很难理解对抗性分类和识别模型漏洞。为了解决这个问题，我们提出了可解释性和透明性驱动检测和转换（IT-DT）框架。它重点关注检测和转换文本对抗示例的可解释性和透明度。IT-DT利用注意力图、综合梯度和模型反馈等技术来实现检测期间的可解释性。这有助于识别有助于对抗性分类的显着特征和扰动词。在转换阶段，IT-DT使用预训练的嵌入和模型反馈来生成受干扰单词的最佳替换。通过找到合适的替代，我们的目标是将对抗性示例转换为非对抗性示例，这些示例与模型的预期行为一致，同时保留文本的含义。通过人类专家的参与来强调透明度。专家审查检测和转化结果并提供反馈，增强决策，尤其是在复杂场景中。该框架生成见解和威胁情报，使分析师能够识别漏洞并提高模型稳健性。全面的实验证明了IT-DT在检测和转化对抗性示例方面的有效性。该方法增强了可解释性、提供透明度，并能够准确识别和成功转换对抗性输入。通过结合技术分析和人类专业知识，IT-DT显着提高了基于转换器的文本分类器抵御对抗攻击的弹性和可信度。



## **44. RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs**

RAGE：适用于越狱LLM的参考感知和集成解码 cs.CL

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.13901v1) [paper-pdf](http://arxiv.org/pdf/2510.13901v1)

**Authors**: Tuan T. Nguyen, John Le, Thai T. Vu, Willy Susilo, Heath Cooper

**Abstract**: Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.

摘要: 大型语言模型（LLM）在不同任务中取得了令人印象深刻的性能，但仍然容易受到绕过安全机制的越狱攻击。我们介绍了RAID-Aware（参考感知和集成解码），这是一个框架，通过制作对抗性后缀来系统性地探索这些弱点，这些后缀在保持流畅性的同时引入受限制的内容。RAID将离散令牌放松为连续嵌入，并通过联合目标对其进行优化，该目标（i）鼓励限制响应，（ii）合并反推感知规则化器以引导激活远离嵌入空间中的拒绝方向，以及（iii）应用一致性项来保持语义一致性和非冗余性。优化后，批评引导的解码过程通过平衡嵌入亲和力与语言模型可能性来将嵌入映射回令牌。这种集成产生的后缀既可以有效绕过防御，而且形式自然。在多个开源LLM上的实验表明，与最近的白盒和黑盒基线相比，与更少的查询和更低的计算成本实现了更高的攻击成功率。这些发现凸显了嵌入空间规范化对于理解和缓解LLM越狱漏洞的重要性。



## **45. SoundnessBench: A Soundness Benchmark for Neural Network Verifiers**

SoundnessBench：神经网络验证者的健全基准 cs.LG

Preprint

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2412.03154v2) [paper-pdf](http://arxiv.org/pdf/2412.03154v2)

**Authors**: Xingjian Zhou, Keyi Shen, Andy Xu, Hongji Xu, Cho-Jui Hsieh, Huan Zhang, Zhouxing Shi

**Abstract**: Neural network (NN) verification aims to formally verify properties of NNs, which is crucial for ensuring the behavior of NN-based models in safety-critical applications. In recent years, the community has developed many NN verifiers and benchmarks to evaluate them. However, existing benchmarks typically lack ground-truth for hard instances where no current verifier can verify the property and no counterexample can be found. This makes it difficult to validate the soundness of a verifier, when it claims verification on such challenging instances that no other verifier can handle. In this work, we develop a new benchmark for NN verification, named "SoundnessBench", specifically for testing the soundness of NN verifiers. SoundnessBench consists of instances with deliberately inserted counterexamples that are hidden from adversarial attacks commonly used to find counterexamples. Thereby, it can identify false verification claims when hidden counterexamples are known to exist. We design a training method to produce NNs with hidden counterexamples and systematically construct our SoundnessBench with instances across various model architectures, activation functions, and input data. We demonstrate that our training effectively produces hidden counterexamples and our SoundnessBench successfully identifies bugs in state-of-the-art NN verifiers. Our code is available at https://github.com/MVP-Harry/SoundnessBench and our benchmark is available at https://huggingface.co/datasets/SoundnessBench/SoundnessBench.

摘要: 神经网络（NN）验证旨在正式验证NN的属性，这对于确保基于NN的模型在安全关键应用中的行为至关重要。近年来，社区开发了许多NN验证器和基准来评估它们。然而，现有的基准测试通常缺乏针对当前验证者可以验证属性且找不到反例的硬实例的基本事实。这使得当验证者声称对没有其他验证者可以处理的具有挑战性的实例进行验证时，很难验证者的合理性。在这项工作中，我们开发了一个新的NN验证基准，名为“SoundnessBench”，专门用于测试NN验证器的可靠性。SoundnessBench由带有故意插入反例的实例组成，这些反例隐藏在通常用于寻找反例的对抗性攻击中。因此，当已知存在隐藏反例时，它可以识别错误的验证声明。我们设计了一种训练方法来生成具有隐藏反例的NN，并使用跨各种模型架构、激活函数和输入数据的实例系统地构建我们的SoundnessBench。我们证明我们的训练有效地产生了隐藏的反例，并且我们的SoundnessBench成功地识别了最先进的NN验证器中的错误。我们的代码可在https://github.com/MVP-Harry/SoundnessBench上获取，我们的基准可在https://huggingface.co/datasets/SoundnessBench/SoundnessBench上获取。



## **46. A Survey of Graph Unlearning**

图形遗忘研究综述 cs.LG

15 page review paper on graph unlearning

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2310.02164v4) [paper-pdf](http://arxiv.org/pdf/2310.02164v4)

**Authors**: Anwar Said, Ngoc N. Tran, Yuying Zhao, Tyler Derr, Mudassir Shabbir, Waseem Abbas, Xenofon Koutsoukos

**Abstract**: Graph unlearning emerges as a crucial advancement in the pursuit of responsible AI, providing the means to remove sensitive data traces from trained models, thereby upholding the \textit{right to be forgotten}. It is evident that graph machine learning exhibits sensitivity to data privacy and adversarial attacks, necessitating the application of graph unlearning techniques to address these concerns effectively. In this comprehensive survey paper, we present the first systematic review of graph unlearning approaches, encompassing a diverse array of methodologies and offering a detailed taxonomy and up-to-date literature overview to facilitate the understanding of researchers new to this field. To ensure clarity, we provide lucid explanations of the fundamental concepts and evaluation measures used in graph unlearning, catering to a broader audience with varying levels of expertise. Delving into potential applications, we explore the versatility of graph unlearning across various domains, including but not limited to social networks, adversarial settings, recommender systems, and resource-constrained environments like the Internet of Things, illustrating its potential impact in safeguarding data privacy and enhancing AI systems' robustness. Finally, we shed light on promising research directions, encouraging further progress and innovation within the domain of graph unlearning. By laying a solid foundation and fostering continued progress, this survey seeks to inspire researchers to further advance the field of graph unlearning, thereby instilling confidence in the ethical growth of AI systems and reinforcing the responsible application of machine learning techniques in various domains.

摘要: 图形取消学习是追求负责任的人工智能的一项关键进步，它提供了从训练模型中删除敏感数据痕迹的方法，从而维护了\textit{被遗忘权}。显然，图机器学习对数据隐私和对抗攻击表现出敏感性，因此需要应用图去学习技术来有效解决这些问题。在这篇全面的调查论文中，我们首次对图形学习方法进行了系统性回顾，涵盖了多种方法，并提供了详细的分类学和最新的文献概述，以促进对该领域新接触的研究人员的理解。为了确保清晰度，我们对图形取消学习中使用的基本概念和评估指标进行了清晰的解释，以迎合具有不同专业知识水平的更广泛受众。通过深入研究潜在的应用，我们探索了图去学习在各个领域的多功能性，包括但不限于社交网络、对抗性环境、推荐系统和物联网等资源受限环境，说明了其在保护数据隐私和增强人工智能系统稳健性方面的潜在影响。最后，我们揭示了有前途的研究方向，鼓励在图去学习领域取得进一步的进步和创新。通过奠定坚实的基础和促进持续进步，这项调查旨在激励研究人员进一步推进图学习领域，从而为人工智能系统的道德发展注入信心，并加强机器学习技术在各个领域的负责任应用。



## **47. KoALA: KL-L0 Adversarial Detector via Label Agreement**

KoALA：通过标签协议的KL-L0对抗检测器 cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12752v1) [paper-pdf](http://arxiv.org/pdf/2510.12752v1)

**Authors**: Siqi Li, Yasser Shoukry

**Abstract**: Deep neural networks are highly susceptible to adversarial attacks, which pose significant risks to security- and safety-critical applications. We present KoALA (KL-L0 Adversarial detection via Label Agreement), a novel, semantics-free adversarial detector that requires no architectural changes or adversarial retraining. KoALA operates on a simple principle: it detects an adversarial attack when class predictions from two complementary similarity metrics disagree. These metrics-KL divergence and an L0-based similarity-are specifically chosen to detect different types of perturbations. The KL divergence metric is sensitive to dense, low-amplitude shifts, while the L0-based similarity is designed for sparse, high-impact changes. We provide a formal proof of correctness for our approach. The only training required is a simple fine-tuning step on a pre-trained image encoder using clean images to ensure the embeddings align well with both metrics. This makes KOALA a lightweight, plug-and-play solution for existing models and various data modalities. Our extensive experiments on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet confirm our theoretical claims. When the theorem's conditions are met, KoALA consistently and effectively detects adversarial examples. On the full test sets, KoALA achieves a precision of 0.94 and a recall of 0.81 on ResNet/CIFAR-10, and a precision of 0.66 and a recall of 0.85 on CLIP/Tiny-ImageNet.

摘要: 深度神经网络极易受到对抗攻击，这对安全和安全关键应用程序构成重大风险。我们提出了KoALA（通过标签协议进行KL-L0对抗性检测），这是一种新颖的、无语义的对抗性检测器，不需要架构更改或对抗性再培训。KoALA的工作原理很简单：当来自两个互补相似性指标的类预测不一致时，它会检测到对抗攻击。这些指标（KL偏差和基于L0的相似性）是专门选择的，以检测不同类型的扰动。KL背离指标对密集、低幅度的漂移敏感，而基于L0的相似性则针对稀疏、高影响的变化而设计。我们为我们的方法提供了正确性的正式证明。唯一需要的训练是使用干净图像对预训练的图像编码器进行简单的微调步骤，以确保嵌入与两个指标良好一致。这使得KOALA成为现有模型和各种数据模式的轻量级、即插即用解决方案。我们对ResNet/CIFAR-10和CLIP/Tiny-ImageNet的广泛实验证实了我们的理论主张。当满足该定理的条件时，KoALA会一致有效地检测对抗性示例。在完整测试集上，KoALA在ResNet/CIFAR-10上的精确度为0.94，召回率为0.81，在CLIP/Tiny-ImageNet上的精确度为0.66，召回率为0.85。



## **48. Towards Robust Artificial Intelligence: Self-Supervised Learning Approach for Out-of-Distribution Detection**

迈向稳健的人工智能：用于分布外检测的自我监督学习方法 cs.AI

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12713v1) [paper-pdf](http://arxiv.org/pdf/2510.12713v1)

**Authors**: Wissam Salhab, Darine Ameyed, Hamid Mcheick, Fehmi Jaafar

**Abstract**: Robustness in AI systems refers to their ability to maintain reliable and accurate performance under various conditions, including out-of-distribution (OOD) samples, adversarial attacks, and environmental changes. This is crucial in safety-critical systems, such as autonomous vehicles, transportation, or healthcare, where malfunctions could have severe consequences. This paper proposes an approach to improve OOD detection without the need of labeled data, thereby increasing the AI systems' robustness. The proposed approach leverages the principles of self-supervised learning, allowing the model to learn useful representations from unlabeled data. Combined with graph-theoretical techniques, this enables the more efficient identification and categorization of OOD samples. Compared to existing state-of-the-art methods, this approach achieved an Area Under the Receiver Operating Characteristic Curve (AUROC) = 0.99.

摘要: 人工智能系统的鲁棒性是指它们在各种条件下保持可靠和准确性能的能力，包括分发外（OOD）样本、对抗性攻击和环境变化。这对于自动驾驶汽车、交通或医疗保健等安全关键系统至关重要，这些系统的故障可能会造成严重后果。本文提出了一种在不需要标记数据的情况下改进OOD检测的方法，从而提高人工智能系统的鲁棒性。所提出的方法利用了自我监督学习的原则，使模型能够从未标记的数据中学习有用的表示。结合图论技术，这使得更有效的识别和分类的OOD样本。与现有的最先进的方法相比，该方法实现了受试者工作特征曲线下面积（AUROC）= 0.99。



## **49. Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain**

Agentland的恶意：深入人工智能供应链后门的兔子洞 cs.CR

27 pages

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.05159v2) [paper-pdf](http://arxiv.org/pdf/2510.05159v2)

**Authors**: Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dj Dvijotham, Alexandre Drouin

**Abstract**: The practice of fine-tuning AI agents on data from their own interactions--such as web browsing or tool use--, while being a strong general recipe for improving agentic capabilities, also introduces a critical security vulnerability within the AI supply chain. In this work, we show that adversaries can easily poison the data collection pipeline to embed hard-to-detect backdoors that are triggerred by specific target phrases, such that when the agent encounters these triggers, it performs an unsafe or malicious action. We formalize and validate three realistic threat models targeting different layers of the supply chain: 1) direct poisoning of fine-tuning data, where an attacker controls a fraction of the training traces; 2) environmental poisoning, where malicious instructions are injected into webpages scraped or tools called while creating training data; and 3) supply chain poisoning, where a pre-backdoored base model is fine-tuned on clean data to improve its agentic capabilities. Our results are stark: by poisoning as few as 2% of the collected traces, an attacker can embed a backdoor causing an agent to leak confidential user information with over 80% success when a specific trigger is present. This vulnerability holds across all three threat models. Furthermore, we demonstrate that prominent safeguards, including two guardrail models and one weight-based defense, fail to detect or prevent the malicious behavior. These findings highlight an urgent threat to agentic AI development and underscore the critical need for rigorous security vetting of data collection processes and end-to-end model supply chains.

摘要: 根据人工智能代理自身交互中的数据（例如网络浏览或工具使用）进行微调的做法虽然是提高代理能力的强大通用配方，但也在人工智能供应链中引入了一个关键的安全漏洞。在这项工作中，我们表明，对手可以很容易地毒害数据收集管道，以嵌入被特定目标短语触发的难以检测的后门，这样当代理遇到这些触发器时，就会执行不安全或恶意的操作。我们形式化并验证了三种针对供应链不同层的现实威胁模型：1）微调数据的直接中毒，其中攻击者控制了一小部分训练痕迹; 2）环境中毒，其中恶意指令被注入到创建训练数据时抓取的网页或调用的工具中; 3）供应链中毒，即根据干净的数据对预先后门的基础模型进行微调，以提高其代理能力。我们的结果很明显：攻击者可以通过毒害仅2%的收集痕迹，嵌入后门，导致代理在存在特定触发器时泄露机密用户信息，成功率超过80%。该漏洞适用于所有三种威胁模型。此外，我们证明，包括两种护栏模型和一种基于重量的防御在内的主要防护措施无法检测或防止恶意行为。这些发现凸显了代理人工智能开发面临的紧迫威胁，并强调了对数据收集流程和端到端模型供应链进行严格安全审查的迫切需要。



## **50. PEAR: Planner-Executor Agent Robustness Benchmark**

PEAR：规划者-执行者代理稳健性基准 cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07505v2) [paper-pdf](http://arxiv.org/pdf/2510.07505v2)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）已成为处理跨不同领域复杂、多步骤任务的强大范式。然而，尽管MAS的能力令人印象深刻，但仍然容易受到对抗操纵。现有的研究通常会检查孤立的攻击表面或特定场景，从而缺乏对MAS漏洞的全面了解。为了弥合这一差距，我们引入了PEAR，这是一个用于系统评估规划者-执行者MAS的实用性和脆弱性的基准。虽然兼容各种MAS体系结构，我们的基准集中在规划者-执行器结构，这是一个实用的和广泛采用的设计。通过大量的实验，我们发现：（1）弱规划器比弱执行器更严重地降低了清洁任务的整体性能;（2）虽然规划器的内存模块是必不可少的，但执行器的内存模块并不影响清洁任务的性能;（3）任务性能和鲁棒性之间存在权衡;以及（4）针对计划者的攻击在误导系统方面特别有效。这些发现提供了可操作的见解，提高MAS的鲁棒性，并奠定了基础，在多智能体设置的原则性防御。



