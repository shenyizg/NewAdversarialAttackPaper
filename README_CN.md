# Latest Adversarial Attack Papers
**update at 2025-07-26 11:30:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Optimal Transport Regularized Divergences: Application to Adversarial Robustness**

最佳运输正规化分歧：对抗稳健性的应用 cs.LG

34 pages, 2 figures

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2309.03791v3) [paper-pdf](http://arxiv.org/pdf/2309.03791v3)

**Authors**: Jeremiah Birrell, Reza Ebrahimi

**Abstract**: We introduce a new class of optimal-transport-regularized divergences, $D^c$, constructed via an infimal convolution between an information divergence, $D$, and an optimal-transport (OT) cost, $C$, and study their use in distributionally robust optimization (DRO). In particular, we propose the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These DRO-based methods are defined by minimizing the maximum expected loss over a $D^c$-neighborhood of the empirical distribution of the training data. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence; the addition of a principled and dynamical adversarial re-weighting on top of adversarial sample transport is a key innovation of $ARMOR_D$. $ARMOR_D$ can be viewed as a generalization of the best-performing loss functions and OT costs in the adversarial training literature; we demonstrate this flexibility by using $ARMOR_D$ to augment the UDR, TRADES, and MART methods and obtain improved performance on CIFAR-10 and CIFAR-100 image recognition. Specifically, augmenting with $ARMOR_D$ leads to 1.9\% and 2.1\% improvement against AutoAttack, a powerful ensemble of adversarial attacks, on CIFAR-10 and CIFAR-100 respectively. To foster reproducibility, we made the code accessible at https://github.com/star-ailab/ARMOR.

摘要: 我们引入了一类新的最优传输正规化分歧$D ' c '，通过信息分歧$D$和最优传输（OT）成本$C$之间的小卷积构建，并研究它们在分布式鲁棒优化（DRO）中的用途。特别是，我们提出了$ARMOR_D$方法作为增强深度学习模型对抗鲁棒性的新颖方法。这些基于DRO的方法是通过最小化训练数据经验分布的$D ' c '附近的最大预期损失来定义的。作为构建对抗性样本的工具，我们的方法允许根据OT成本传输样本，并根据信息差异重新加权;在对抗性样本传输之上添加有原则且动态的对抗性重新加权是$ARMOR_D$的关键创新。$ARMOR_D$可以被视为对抗训练文献中表现最佳的损失函数和OT成本的概括;我们通过使用$ARMOR_D$来增强UDR、TRADES和MART方法并在CIFAR-10和CIFAR-100图像识别上获得更好的性能来证明这种灵活性。具体来说，在CIFAR-10和CIFAR-100上，使用$ARMOR_D$进行增强后，AutoAttack（一种强大的对抗性攻击集合）分别提高了1.9%和2.1%。为了提高可重复性，我们在https://github.com/star-ailab/ARMOR上提供了该代码。



## **2. GCC-Spam: Spam Detection via GAN, Contrastive Learning, and Character Similarity Networks**

GCC-Spam：通过GAN，对比学习和字符相似性网络进行垃圾邮件检测 cs.LG

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.14679v2) [paper-pdf](http://arxiv.org/pdf/2507.14679v2)

**Authors**: Zhijie Wang, Zixin Xu, Zhiyuan Pan

**Abstract**: The exponential growth of spam text on the Internet necessitates robust detection mechanisms to mitigate risks such as information leakage and social instability. This work addresses two principal challenges: adversarial strategies employed by spammers and the scarcity of labeled data. We propose a novel spam-text detection framework GCC-Spam, which integrates three core innovations. First, a character similarity network captures orthographic and phonetic features to counter character-obfuscation attacks and furthermore produces sentence embeddings for downstream classification. Second, contrastive learning enhances discriminability by optimizing the latent-space distance between spam and normal texts. Third, a Generative Adversarial Network (GAN) generates realistic pseudo-spam samples to alleviate data scarcity while improving model robustness and classification accuracy. Extensive experiments on real-world datasets demonstrate that our model outperforms baseline approaches, achieving higher detection rates with significantly fewer labeled examples.

摘要: 互联网上垃圾文本呈指数级增长，需要强大的检测机制来减轻信息泄露和社会不稳定等风险。这项工作解决了两个主要挑战：垃圾邮件发送者采用的对抗策略和标记数据的稀缺性。我们提出了一种新颖的垃圾邮件文本检测框架GCC-Spam，该框架集成了三项核心创新。首先，字符相似性网络捕获正音特征以对抗字符混淆攻击，并进一步生成用于下游分类的句子嵌入。其次，对比学习通过优化垃圾邮件和正常文本之间的潜在空间距离来增强区分性。第三，生成对抗网络（GAN）生成真实的伪垃圾邮件样本，以缓解数据稀缺性，同时提高模型稳健性和分类准确性。对现实世界数据集的广泛实验表明，我们的模型优于基线方法，以显着更少的标记示例实现了更高的检测率。



## **3. Reinforced Embodied Active Defense: Exploiting Adaptive Interaction for Robust Visual Perception in Adversarial 3D Environments**

强化的协同主动防御：利用自适应交互在对抗性3D环境中实现稳健的视觉感知 cs.CV

arXiv admin note: text overlap with arXiv:2404.00540

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18484v1) [paper-pdf](http://arxiv.org/pdf/2507.18484v1)

**Authors**: Xiao Yang, Lingxuan Wu, Lizhong Wang, Chengyang Ying, Hang Su, Jun Zhu

**Abstract**: Adversarial attacks in 3D environments have emerged as a critical threat to the reliability of visual perception systems, particularly in safety-sensitive applications such as identity verification and autonomous driving. These attacks employ adversarial patches and 3D objects to manipulate deep neural network (DNN) predictions by exploiting vulnerabilities within complex scenes. Existing defense mechanisms, such as adversarial training and purification, primarily employ passive strategies to enhance robustness. However, these approaches often rely on pre-defined assumptions about adversarial tactics, limiting their adaptability in dynamic 3D settings. To address these challenges, we introduce Reinforced Embodied Active Defense (Rein-EAD), a proactive defense framework that leverages adaptive exploration and interaction with the environment to improve perception robustness in 3D adversarial contexts. By implementing a multi-step objective that balances immediate prediction accuracy with predictive entropy minimization, Rein-EAD optimizes defense strategies over a multi-step horizon. Additionally, Rein-EAD involves an uncertainty-oriented reward-shaping mechanism that facilitates efficient policy updates, thereby reducing computational overhead and supporting real-world applicability without the need for differentiable environments. Comprehensive experiments validate the effectiveness of Rein-EAD, demonstrating a substantial reduction in attack success rates while preserving standard accuracy across diverse tasks. Notably, Rein-EAD exhibits robust generalization to unseen and adaptive attacks, making it suitable for real-world complex tasks, including 3D object classification, face recognition and autonomous driving.

摘要: 3D环境中的对抗性攻击已经成为视觉感知系统可靠性的关键威胁，特别是在身份验证和自动驾驶等安全敏感型应用中。这些攻击使用对抗补丁和3D对象，通过利用复杂场景中的漏洞来操纵深度神经网络（DNN）预测。现有的防御机制，如对抗训练和净化，主要采用被动策略来增强鲁棒性。然而，这些方法通常依赖于预先定义的对抗策略假设，限制了它们在动态3D环境中的适应性。为了应对这些挑战，我们引入了加强型主动防御（Rein-EAD），这是一种主动防御框架，利用自适应探索和与环境的交互来提高3D对抗环境中的感知稳健性。通过实施平衡即时预测准确性与预测性最小化的多步骤目标，Rein-EAD在多步骤范围内优化了防御策略。此外，Rein-EAD涉及一种面向不确定性的奖励塑造机制，该机制有助于高效的策略更新，从而减少计算负担并支持现实世界的适用性，而无需差异化环境。全面的实验验证了Rein-EAD的有效性，证明攻击成功率大幅降低，同时在不同任务中保持标准准确性。值得注意的是，Rein-EAD对不可见和自适应攻击具有强大的概括性，使其适合现实世界的复杂任务，包括3D对象分类、面部识别和自动驾驶。



## **4. Revisiting Physically Realizable Adversarial Object Attack against LiDAR-based Detection: Clarifying Problem Formulation and Experimental Protocols**

重新审视针对基于LiTAR的检测的物理可实现对抗对象攻击：澄清问题制定和实验协议 cs.CV

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18457v1) [paper-pdf](http://arxiv.org/pdf/2507.18457v1)

**Authors**: Luo Cheng, Hanwei Zhang, Lijun Zhang, Holger Hermanns

**Abstract**: Adversarial robustness in LiDAR-based 3D object detection is a critical research area due to its widespread application in real-world scenarios. While many digital attacks manipulate point clouds or meshes, they often lack physical realizability, limiting their practical impact. Physical adversarial object attacks remain underexplored and suffer from poor reproducibility due to inconsistent setups and hardware differences. To address this, we propose a device-agnostic, standardized framework that abstracts key elements of physical adversarial object attacks, supports diverse methods, and provides open-source code with benchmarking protocols in simulation and real-world settings. Our framework enables fair comparison, accelerates research, and is validated by successfully transferring simulated attacks to a physical LiDAR system. Beyond the framework, we offer insights into factors influencing attack success and advance understanding of adversarial robustness in real-world LiDAR perception.

摘要: 基于LiDART的3D对象检测中的对抗鲁棒性因其在现实世界场景中的广泛应用而成为一个关键的研究领域。虽然许多数字攻击操纵点云或网格，但它们通常缺乏物理可实现性，限制了其实际影响。物理对抗对象攻击仍然未充分研究，并且由于设置不一致和硬件差异而重现性较差。为了解决这个问题，我们提出了一个设备不可知的标准化框架，该框架抽象物理对抗对象攻击的关键元素，支持多种方法，并在模拟和现实世界环境中提供具有基准协议的开源代码。我们的框架可以进行公平的比较，加速研究，并通过成功地将模拟攻击转移到物理LiDAR系统进行验证。除了框架之外，我们还提供了对影响攻击成功的因素的见解，并进一步了解了现实世界LiDAR感知中的对抗鲁棒性。



## **5. On Reconstructing Training Data From Bayesian Posteriors and Trained Models**

关于从Bayesian后验和训练模型重建训练数据 stat.ML

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18372v1) [paper-pdf](http://arxiv.org/pdf/2507.18372v1)

**Authors**: George Wynne

**Abstract**: Publicly releasing the specification of a model with its trained parameters means an adversary can attempt to reconstruct information about the training data via training data reconstruction attacks, a major vulnerability of modern machine learning methods. This paper makes three primary contributions: establishing a mathematical framework to express the problem, characterising the features of the training data that are vulnerable via a maximum mean discrepancy equivalance and outlining a score matching framework for reconstructing data in both Bayesian and non-Bayesian models, the former is a first in the literature.

摘要: 公开发布具有训练参数的模型规范意味着对手可以尝试通过训练数据重建攻击来重建有关训练数据的信息，这是现代机器学习方法的一个主要弱点。本文做出了三个主要贡献：建立一个数学框架来表达问题，通过最大平均差异等效来描述训练数据中脆弱的特征，并概述了一个用于在Bayesian和非Bayesian模型中重建数据的得分匹配框架，前者是文献中的首次。



## **6. Data Transmission over a Bosonic Arbitrarily Varying Quantum Channel**

玻色子随机变化量子通道上的数据传输 quant-ph

8 pages, no figures

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18259v1) [paper-pdf](http://arxiv.org/pdf/2507.18259v1)

**Authors**: Janis Nötzel, Florian Seitz

**Abstract**: Arbitrarily varying channels offer a powerful framework for analyzing the robustness of quantum communication systems, especially for classical-quantum models, where the analysis displays strengths or weaknesses of specific signal constellations under generic attacks. In this work, we provide a coding theorem for a large class of practically relevant arbitrarily varying channel models. Namely, we give an explicit capacity formula for the lossy bosonic channel subject to semi-classical attacks, where an adversary injects semi-classical states into the transmission line. Mathematically, this is modeled via a beam-splitter setup, with transmitter and jammer controlling different input ports and the receiver observing one output port. We show how a recently conjectured new quantum entropy power inequality relates to our capacity formula.

摘要: 不同的通道为分析量子通信系统的稳健性提供了一个强大的框架，特别是对于经典量子模型，其中分析显示了特定信号星座在一般攻击下的优点或缺点。在这项工作中，我们为一大类实际相关的任意变化的通道模型提供了编码定理。也就是说，我们给出了受到半经典攻击的有损耗玻色子通道的显式容量公式，其中对手将半经典状态注入传输线。从数学上讲，这是通过分束器设置进行建模的，发射机和干扰机控制不同的输入端口，接收机观察一个输出端口。我们展示了最近假设的新量子熵功率不等式如何与我们的容量公式相关。



## **7. Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection**

使用GMTA保护RAG管道：一种基于对象的掩蔽令牌概率方法，用于有毒文档检测 cs.CL

18 pages, accepted to ACL Findings 2025

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18202v1) [paper-pdf](http://arxiv.org/pdf/2507.18202v1)

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings.

摘要: 检索增强生成（RAG）通过提供外部知识以提供准确和最新的响应来增强大型语言模型（LLM）。然而，这种对外部来源的依赖暴露了安全风险，攻击者可以将有毒文档注入知识库，以引导生成过程转向有害或误导性的输出。在本文中，我们提出了基于对象的掩蔽令牌概率（GMTA），这是一种新型防御方法，用于检测和过滤敌对精心设计的文档。具体来说，GMTA通过检查检索器相似性函数的梯度来识别高影响力代币。然后，这些关键令牌被屏蔽，并通过屏蔽语言模型（MLM）检查它们的概率。由于注入的令牌通常表现出明显较低的被屏蔽令牌概率，这使GMTA能够轻松检测恶意文档并实现高精度过滤。实验表明，GMat能够消除超过90%的有毒内容，同时保留相关文档，从而在不同数据集和对抗性设置中保持稳健的检索和生成性能。



## **8. Policy Disruption in Reinforcement Learning:Adversarial Attack with Large Language Models and Critical State Identification**

强化学习中的政策破坏：大型语言模型和关键状态识别的对抗性攻击 cs.LG

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18113v1) [paper-pdf](http://arxiv.org/pdf/2507.18113v1)

**Authors**: Junyong Jiang, Buwei Tian, Chenxing Xu, Songze Li, Lu Dong

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in fields like robotics and autonomous driving, but adversarial attacks designed to mislead RL systems remain challenging. Existing approaches often rely on modifying the environment or policy, limiting their practicality. This paper proposes an adversarial attack method in which existing agents in the environment guide the target policy to output suboptimal actions without altering the environment. We propose a reward iteration optimization framework that leverages large language models (LLMs) to generate adversarial rewards explicitly tailored to the vulnerabilities of the target agent, thereby enhancing the effectiveness of inducing the target agent toward suboptimal decision-making. Additionally, a critical state identification algorithm is designed to pinpoint the target agent's most vulnerable states, where suboptimal behavior from the victim leads to significant degradation in overall performance. Experimental results in diverse environments demonstrate the superiority of our method over existing approaches.

摘要: 强化学习（RL）在机器人和自动驾驶等领域取得了显着的成功，但旨在误导强化学习系统的对抗性攻击仍然具有挑战性。现有的方法通常依赖于修改环境或政策，限制了其实用性。本文提出了一种对抗攻击方法，其中环境中现有的代理引导目标策略在不改变环境的情况下输出次优动作。我们提出了一个奖励迭代优化框架，该框架利用大型语言模型（LLM）来生成明确针对目标代理的脆弱性定制的对抗奖励，从而提高诱导目标代理进行次优决策的有效性。此外，关键状态识别算法旨在确定目标代理最脆弱的状态，其中受害者的次优行为导致总体性能显着下降。不同环境中的实验结果证明了我们的方法优于现有方法。



## **9. RECALLED: An Unbounded Resource Consumption Attack on Large Vision-Language Models**

重新命名：对大型视觉语言模型的无限资源消耗攻击 cs.CR

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18053v1) [paper-pdf](http://arxiv.org/pdf/2507.18053v1)

**Authors**: Haoran Gao, Yuanhe Zhang, Zhenhong Zhou, Lei Jiang, Fanyu Meng, Yujia Xiao, Kun Wang, Yang Liu, Junlan Feng

**Abstract**: Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs). With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). However, existing red-teaming studies have largely overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs. To address this gap, we propose RECALLED (\textbf{RE}source \textbf{C}onsumption \textbf{A}ttack on \textbf{L}arge Vision-\textbf{L}anguag\textbf{E} Mo\textbf{D}els), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming. First, we present \textit{Vision Guided Optimization}, a fine-grained pixel-level optimization, to obtain \textit{Output Recall} adversarial perturbations, which can induce repeating output. Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs. Additionally, we introduce \textit{Multi-Objective Parallel Losses} to generate universal attack templates and resolve optimization conflicts when intending to implement parallel attacks. Empirical results demonstrate that RECALLED increases service response latency by over 26 $\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. Our study exposes security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate future defense development against RCAs.

摘要: 资源消耗攻击（RCA）已成为大型语言模型（LLM）部署的重大威胁。随着视觉模式的集成，额外的攻击载体加剧了大型视觉语言模型（LVLM）中RCA的风险。然而，现有的红色团队研究在很大程度上忽视了视觉输入作为潜在攻击表面，导致针对LVLM中RCA的缓解策略不足。为了解决这一差距，我们提出了RECALLED（\textBF{RE}source \textBF{C}onsumption \textBF{A}ttack on \textBF {L}arge Vision-\textBF {L}anguag\textBF{E} Mo\textBF{D}els），这是第一种利用视觉模式触发无界RCA红色分组的方法。首先，我们提出了\textit{Vision Guided Optimism}，一种细粒度像素级优化，以获得\textit{Exit Recall}对抗性扰动，这可能会导致重复输出。然后，我们将扰动注入视觉输入中，触发无限世代来实现RCA的目标。此外，我们还引入了\texttit {Multi-Observer并行损失}来生成通用攻击模板并在打算实施并行攻击时解决优化冲突。经验结果表明，RECALLED将服务响应延迟增加了超过26 $\uparrow$，导致图形处理器利用率和内存消耗额外增加20%。我们的研究揭示了LVLM中的安全漏洞，并建立了一个红色团队框架，可以促进未来针对RCA的防御开发。



## **10. Your ATs to Ts: MITRE ATT&CK Attack Technique to P-SSCRM Task Mapping**

您的AT到TS：MITRE ATT & CK攻击技术到P-SSCRM任务映射 cs.SE

Mapping generated from: arXiv:2503.12192

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18037v1) [paper-pdf](http://arxiv.org/pdf/2507.18037v1)

**Authors**: Sivana Hamer, Jacob Bowen, Md Nazmul Haque, Chris Madden, Laurie Williams

**Abstract**: The MITRE Adversarial Tactics, Techniques and Common Knowledge (MITRE ATT&CK) Attack Technique to Proactive Software Supply Chain Risk Management Framework (P-SSCRM) Task mapping described in this document helps software organizations to determine how different tasks mitigate the attack techniques of software supply chain attacks. The mapping was created through four independent strategies to find agreed-upon mappings. Because each P-SSCRM task is mapped to one or more tasks from the 10 frameworks, the mapping we provide is also a mapping between MITRE ATT&CK and other prominent government and industry frameworks.

摘要: MITRE对抗战术、技术和常识（MITRE ATT & CK）攻击技术到主动软件供应链风险管理框架（P-SSCRM）本文档中描述的任务映射帮助软件组织确定不同任务如何缓解软件供应链攻击的攻击技术。该映射是通过四个独立的策略创建的，以寻找商定的映射。由于每个P-SSCRM任务都映射到10个框架中的一个或多个任务，因此我们提供的映射也是MITRE ATT & CK与其他著名政府和行业框架之间的映射。



## **11. Evaluating the Performance of AI Text Detectors, Few-Shot and Chain-of-Thought Prompting Using DeepSeek Generated Text**

使用DeepSeek生成的文本评估AI文本检测器、Few-shot和思想链打印的性能 cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17944v1) [paper-pdf](http://arxiv.org/pdf/2507.17944v1)

**Authors**: Hulayyil Alshammari, Praveen Rao

**Abstract**: Large language models (LLMs) have rapidly transformed the creation of written materials. LLMs have led to questions about writing integrity, thereby driving the creation of artificial intelligence (AI) detection technologies. Adversarial attacks, such as standard and humanized paraphrasing, inhibit detectors' ability to detect machine-generated text. Previous studies have mainly focused on ChatGPT and other well-known LLMs and have shown varying accuracy across detectors. However, there is a clear gap in the literature about DeepSeek, a recently published LLM. Therefore, in this work, we investigate whether six generally accessible AI detection tools -- AI Text Classifier, Content Detector AI, Copyleaks, QuillBot, GPT-2, and GPTZero -- can consistently recognize text generated by DeepSeek. The detectors were exposed to the aforementioned adversarial attacks. We also considered DeepSeek as a detector by performing few-shot prompting and chain-of-thought reasoning (CoT) for classifying AI and human-written text. We collected 49 human-authored question-answer pairs from before the LLM era and generated matching responses using DeepSeek-v3, producing 49 AI-generated samples. Then, we applied adversarial techniques such as paraphrasing and humanizing to add 196 more samples. These were used to challenge detector robustness and assess accuracy impact. While QuillBot and Copyleaks showed near-perfect performance on original and paraphrased DeepSeek text, others -- particularly AI Text Classifier and GPT-2 -- showed inconsistent results. The most effective attack was humanization, reducing accuracy to 71% for Copyleaks, 58% for QuillBot, and 52% for GPTZero. Few-shot and CoT prompting showed high accuracy, with the best five-shot result misclassifying only one of 49 samples (AI recall 96%, human recall 100%).

摘要: 大型语言模型（LLM）迅速改变了书面材料的创建。LLM引发了有关写作完整性的质疑，从而推动了人工智能（AI）检测技术的创建。对抗性攻击，例如标准和人性化的重述，会抑制检测器检测机器生成文本的能力。之前的研究主要集中在ChatGPT和其他知名的LLM上，并表明不同探测器的准确性各不相同。然而，关于DeepSeek（最近出版的法学硕士）的文献中存在明显的空白。因此，在这项工作中，我们调查了六种通用的人工智能检测工具-- AI文本分类器、内容检测器AI、Copyleaks、QuillBot、GPT-2和GPTZero --是否能够一致地识别DeepSeek生成的文本。探测器遭受了上述对抗性攻击。我们还将DeepSeek视为一个检测器，通过执行少量提示和思维链推理（CoT）来对人工智能和人类书写的文本进行分类。我们收集了LLM时代之前的49个人类创作的问答对，并使用DeepSeek-v3生成匹配响应，生成了49个人工智能生成的样本。然后，我们应用了解释和人性化等对抗技术，添加了196个样本。这些用于挑战检测器的稳健性并评估准确性影响。虽然QuillBot和Copyleaks在原始和转述的DeepSeek文本上表现出近乎完美的性能，但其他文本--尤其是AI文本分类器和GPT-2 --却表现出不一致的结果。最有效的攻击是人性化，将Copyleaks的准确性降低到71%，QuillBot的准确性降低到58%，GPTZero的准确性降低到52%。Fe-shot和CoT提示显示出很高的准确性，最好的五次shot结果仅错误分类了49个样本中的一个（人工智能召回率96%，人类召回率100%）。



## **12. Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation**

鲍勃的五彩纸屑：音乐和视频生成中的语音同步攻击 cs.SD

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17937v1) [paper-pdf](http://arxiv.org/pdf/2507.17937v1)

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr

**Abstract**: Lyrics-to-Song (LS2) generation models promise end-to-end music synthesis from text, yet their vulnerability to training data memorization remains underexplored. We introduce Adversarial PhoneTic Prompting (APT), a novel attack where lyrics are semantically altered while preserving their acoustic structure through homophonic substitutions (e.g., Eminem's famous "mom's spaghetti" $\rightarrow$ "Bob's confetti"). Despite these distortions, we uncover a powerful form of sub-lexical memorization: models like SUNO and YuE regenerate outputs strikingly similar to known training content, achieving high similarity across audio-domain metrics, including CLAP, AudioJudge, and CoverID. This vulnerability persists across multiple languages and genres. More surprisingly, we discover that phoneme-altered lyrics alone can trigger visual memorization in text-to-video models. When prompted with phonetically modified lyrics from Lose Yourself, Veo 3 reconstructs visual elements from the original music video -- including character appearance and scene composition -- despite no visual cues in the prompt. We term this phenomenon phonetic-to-visual regurgitation. Together, these findings expose a critical vulnerability in transcript-conditioned multimodal generation: phonetic prompting alone can unlock memorized audiovisual content, raising urgent questions about copyright, safety, and content provenance in modern generative systems. Example generations are available on our demo page (jrohsc.github.io/music_attack/).

摘要: 歌词到歌曲（LS 2）生成模型承诺从文本进行端到端音乐合成，但它们对训练数据记忆的脆弱性仍然没有得到充分的研究。我们引入了对抗音素插入（APT），这是一种新颖的攻击，其中歌词在语义上被改变，同时通过谐音替换保留其声学结构（例如，阿姆著名的“妈妈的意大利面”$\rightarrow$“鲍勃的五彩纸屑”）。尽管存在这些扭曲，我们还是发现了一种强大的亚词汇记忆形式：SUNO和YuE等模型重新生成与已知训练内容惊人相似的输出，从而在音频领域指标（包括CLAP、AudioJudge和CoverID）之间实现了高度相似性。此漏洞在多种语言和流派中持续存在。更令人惊讶的是，我们发现音素改变的歌词本身就可以触发文本到视频模型中的视觉记忆。当提示《Lose Yourself》中经过语音修改的歌词时，Veo 3会从原始音乐视频中重建视觉元素--包括角色外观和场景构成--尽管提示中没有视觉提示。我们将这种现象称为语音到视觉的回流。总之，这些研究结果暴露了一个关键的脆弱性，在成绩单条件下的多模态生成：语音提示单独可以解锁记忆的视听内容，提出了迫切的问题，在现代生成系统的版权，安全和内容出处。示例生成可以在我们的演示页面（jrohsc.github.io/music_attack/）上找到。



## **13. From Seed to Harvest: Augmenting Human Creativity with AI for Red-teaming Text-to-Image Models**

从种子到收获：利用人工智能增强人类创造力，实现红色团队文本到图像模型 cs.LG

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17922v1) [paper-pdf](http://arxiv.org/pdf/2507.17922v1)

**Authors**: Jessica Quaye, Charvi Rastogi, Alicia Parrish, Oana Inel, Minsuk Kahng, Lora Aroyo, Vijay Janapa Reddi

**Abstract**: Text-to-image (T2I) models have become prevalent across numerous applications, making their robust evaluation against adversarial attacks a critical priority. Continuous access to new and challenging adversarial prompts across diverse domains is essential for stress-testing these models for resilience against novel attacks from multiple vectors. Current techniques for generating such prompts are either entirely authored by humans or synthetically generated. On the one hand, datasets of human-crafted adversarial prompts are often too small in size and imbalanced in their cultural and contextual representation. On the other hand, datasets of synthetically-generated prompts achieve scale, but typically lack the realistic nuances and creative adversarial strategies found in human-crafted prompts. To combine the strengths of both human and machine approaches, we propose Seed2Harvest, a hybrid red-teaming method for guided expansion of culturally diverse, human-crafted adversarial prompt seeds. The resulting prompts preserve the characteristics and attack patterns of human prompts while maintaining comparable average attack success rates (0.31 NudeNet, 0.36 SD NSFW, 0.12 Q16). Our expanded dataset achieves substantially higher diversity with 535 unique geographic locations and a Shannon entropy of 7.48, compared to 58 locations and 5.28 entropy in the original dataset. Our work demonstrates the importance of human-machine collaboration in leveraging human creativity and machine computational capacity to achieve comprehensive, scalable red-teaming for continuous T2I model safety evaluation.

摘要: 文本到图像（T2 I）模型已在众多应用程序中流行起来，因此其针对对抗性攻击的稳健评估成为关键优先事项。持续访问不同领域的新的、具有挑战性的对抗提示对于压力测试这些模型抵御来自多个载体的新型攻击的弹性至关重要。当前用于生成此类提示的技术要么完全由人类创作，要么合成生成。一方面，人为设计的对抗提示数据集通常规模太小，并且其文化和上下文表示不平衡。另一方面，合成生成的提示数据集实现了规模，但通常缺乏人工制作的提示中的现实细微差别和创造性的对抗策略。为了结合人类和机器方法的优势，我们提出了Seed 2Harvest，这是一种混合红色团队方法，用于引导扩展文化多样性、人为制作的对抗性提示种子。生成的提示保留了人类提示的特征和攻击模式，同时保持相当的平均攻击成功率（0.31 NudeNet、0.36 SD NSFW、0.12 Q16）。我们扩展的数据集实现了更高的多样性，具有535个独特地理位置和7.48的香农信息量，而原始数据集中有58个位置和5.28的信息量。我们的工作证明了人机协作在利用人类创造力和机器计算能力实现全面、可扩展的红色团队以持续T2 I模型安全评估方面的重要性。



## **14. Trusted Data Fusion, Multi-Agent Autonomy, Autonomous Vehicles**

可信数据融合、多智能体自主、自动驾驶汽车 eess.SY

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17875v1) [paper-pdf](http://arxiv.org/pdf/2507.17875v1)

**Authors**: R. Spencer Hallyburton, Miroslav Pajic

**Abstract**: Multi-agent collaboration enhances situational awareness in intelligence, surveillance, and reconnaissance (ISR) missions. Ad hoc networks of unmanned aerial vehicles (UAVs) allow for real-time data sharing, but they face security challenges due to their decentralized nature, making them vulnerable to cyber-physical attacks. This paper introduces a trust-based framework for assured sensor fusion in distributed multi-agent networks, utilizing a hidden Markov model (HMM)-based approach to estimate the trustworthiness of agents and their provided information in a decentralized fashion. Trust-informed data fusion prioritizes fusing data from reliable sources, enhancing resilience and accuracy in contested environments. To evaluate the assured sensor fusion under attacks on system/mission sensing, we present a novel multi-agent aerial dataset built from the Unreal Engine simulator. We demonstrate through case studies improved ISR performance and an ability to detect malicious actors in adversarial settings.

摘要: 多代理协作增强了情报、监视和侦察（ZR）任务中的态势感知。无人机（UFO）的自组织网络允许实时数据共享，但由于其去中心化性质，它们面临着安全挑战，使它们容易受到网络物理攻击。本文介绍了一种基于信任的分布式多代理网络中保证传感器融合框架，利用基于隐马尔科夫模型（Markov）的方法以分散的方式估计代理及其提供的信息的可信度。基于信任的数据融合优先考虑融合来自可靠来源的数据，增强有争议环境中的弹性和准确性。为了评估系统/任务感知攻击下的有保证的传感器融合，我们提出了一种新的多智能体航空数据集建立从虚幻引擎模拟器。我们通过案例研究展示了改善的ZR性能以及在对抗环境中检测恶意行为者的能力。



## **15. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的从弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2401.17256v5) [paper-pdf](http://arxiv.org/pdf/2401.17256v5)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **16. Constructing Optimal Noise Channels for Enhanced Robustness in Quantum Machine Learning**

构建最佳噪音通道以增强量子机器学习的鲁棒性 quant-ph

QML technical track at IEEE QCE 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2404.16417v2) [paper-pdf](http://arxiv.org/pdf/2404.16417v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: With the rapid advancement of Quantum Machine Learning (QML), the critical need to enhance security measures against adversarial attacks and protect QML models becomes increasingly evident. In this work, we outline the connection between quantum noise channels and differential privacy (DP), by constructing a family of noise channels which are inherently $\epsilon$-DP: $(\alpha, \gamma)$-channels. Through this approach, we successfully replicate the $\epsilon$-DP bounds observed for depolarizing and random rotation channels, thereby affirming the broad generality of our framework. Additionally, we use a semi-definite program to construct an optimally robust channel. In a small-scale experimental evaluation, we demonstrate the benefits of using our optimal noise channel over depolarizing noise, particularly in enhancing adversarial accuracy. Moreover, we assess how the variables $\alpha$ and $\gamma$ affect the certifiable robustness and investigate how different encoding methods impact the classifier's robustness.

摘要: 随着量子机器学习（QML）的快速发展，增强对抗性攻击的安全措施和保护QML模型的迫切需要变得越来越明显。在这项工作中，我们概述了量子噪声通道和差分隐私（DP）之间的联系，通过构建一个家庭的噪声通道，这是固有的$\alpha，\gamma $-DP：$（\alpha，\gamma）$-通道。通过这种方法，我们成功地复制了$\N $-DP界观察到的去极化和随机旋转通道，从而肯定了我们的框架的广泛的一般性。此外，我们使用一个半定规划，以构建一个最佳的鲁棒信道。在一个小规模的实验评估中，我们展示了使用我们的最佳噪声通道去极化噪声的好处，特别是在提高对抗精度。此外，我们评估变量$\alpha$和$\gamma$如何影响可证明的鲁棒性，并研究不同的编码方法如何影响分类器的鲁棒性。



## **17. Boosting Ray Search Procedure of Hard-label Attacks with Transfer-based Priors**

增强具有基于传输的先验的硬标签攻击的Ray搜索程序 cs.CV

Published at ICLR 2025 (Spotlight paper)

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17577v1) [paper-pdf](http://arxiv.org/pdf/2507.17577v1)

**Authors**: Chen Ma, Xinjie Xu, Shuyu Cheng, Qi Xuan

**Abstract**: One of the most practical and challenging types of black-box adversarial attacks is the hard-label attack, where only the top-1 predicted label is available. One effective approach is to search for the optimal ray direction from the benign image that minimizes the $\ell_p$-norm distance to the adversarial region. The unique advantage of this approach is that it transforms the hard-label attack into a continuous optimization problem. The objective function value is the ray's radius, which can be obtained via binary search at a high query cost. Existing methods use a "sign trick" in gradient estimation to reduce the number of queries. In this paper, we theoretically analyze the quality of this gradient estimation and propose a novel prior-guided approach to improve ray search efficiency both theoretically and empirically. Specifically, we utilize the transfer-based priors from surrogate models, and our gradient estimators appropriately integrate them by approximating the projection of the true gradient onto the subspace spanned by these priors and random directions, in a query-efficient manner. We theoretically derive the expected cosine similarities between the obtained gradient estimators and the true gradient, and demonstrate the improvement achieved by incorporating priors. Extensive experiments on the ImageNet and CIFAR-10 datasets show that our approach significantly outperforms 11 state-of-the-art methods in terms of query efficiency.

摘要: 最实用、最具挑战性的黑匣子对抗攻击类型之一是硬标签攻击，其中只有前1名的预测标签可用。一种有效的方法是从良性图像中搜索最佳射线方向，以最小化与对抗区域的$\ell_p$-norm距离。这种方法的独特优势在于，它将硬标签攻击转化为持续优化问题。目标函数值是射线的半径，可以通过二分搜索以很高的查询成本获得。现有方法在梯度估计中使用“符号技巧”来减少查询数量。本文从理论上分析了这种梯度估计的质量，并提出了一种新型的先验引导方法来从理论上和经验上提高射线搜索效率。具体来说，我们利用来自代理模型的基于转移的先验，我们的梯度估计器通过以查询高效的方式将真实梯度的投影逼近到由这些先验和随机方向跨越的子空间上来适当地集成它们。我们从理论上推导出所获得的梯度估计量和真实梯度之间的预期Cosin相似性，并证明通过合并先验来实现的改进。对ImageNet和CIFAR-10数据集的大量实验表明，我们的方法在查询效率方面显着优于11种最先进的方法。



## **18. An h-space Based Adversarial Attack for Protection Against Few-shot Personalization**

一种基于h空间的对抗攻击，用于防止少镜头个性化 cs.CV

32 pages, 15 figures. Accepted by ACM Multimedia 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17554v1) [paper-pdf](http://arxiv.org/pdf/2507.17554v1)

**Authors**: Xide Xu, Sandesh Kamath, Muhammad Atif Butt, Bogdan Raducanu

**Abstract**: The versatility of diffusion models in generating customized images from few samples raises significant privacy concerns, particularly regarding unauthorized modifications of private content. This concerning issue has renewed the efforts in developing protection mechanisms based on adversarial attacks, which generate effective perturbations to poison diffusion models. Our work is motivated by the observation that these models exhibit a high degree of abstraction within their semantic latent space (`h-space'), which encodes critical high-level features for generating coherent and meaningful content. In this paper, we propose a novel anti-customization approach, called HAAD (h-space based Adversarial Attack for Diffusion models), that leverages adversarial attacks to craft perturbations based on the h-space that can efficiently degrade the image generation process. Building upon HAAD, we further introduce a more efficient variant, HAAD-KV, that constructs perturbations solely based on the KV parameters of the h-space. This strategy offers a stronger protection, that is computationally less expensive. Despite their simplicity, our methods outperform state-of-the-art adversarial attacks, highlighting their effectiveness.

摘要: 扩散模型的多功能性，在生成定制的图像从几个样本提出了显着的隐私问题，特别是关于未经授权的修改私人内容。这一令人关注的问题重新致力于开发基于对抗性攻击的保护机制，这对毒药扩散模型产生了有效的扰动。我们的工作是出于观察，这些模型表现出高度的抽象在其语义的潜在空间（“h-空间”），编码关键的高层次的功能，产生连贯和有意义的内容。在本文中，我们提出了一种新的反定制方法，称为HAAD（基于h空间的对抗性攻击扩散模型），它利用对抗性攻击来制作基于h空间的扰动，可以有效地降低图像生成过程。在HAAD的基础上，我们进一步引入了一种更有效的变体HAAD-KV，它仅根据h空间的KV参数构建扰动。该策略提供了更强的保护，计算成本更低。尽管它们很简单，但我们的方法优于最先进的对抗攻击，凸显了它们的有效性。



## **19. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

MEF：一个用于评估黑箱大语言模型脆弱性的能力感知多重加密框架 cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2505.23404v4) [paper-pdf](http://arxiv.org/pdf/2505.23404v4)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Based on our experiments, we found that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the attacked LLM. Building on this insight, we propose a capability-aware Multi-Encryption Framework (MEF) for evaluating vulnerabilities in black-box LLMs. Specifically, MEF first categorizes the comprehension ability level of the LLM, then applies different strategies accordingly: For models with limited comprehension ability, MEF adopts the Fu+En1 strategy, which integrates layered semantic mutations with an encryption technique, more effectively contributing to evasion of the LLM's defenses at the input and inference stages. For models with strong comprehension ability, MEF uses a more complex Fu+En1+En2 strategy, in which additional dual-ended encryption techniques are applied to the LLM's responses, further contributing to evasion of the LLM's defenses at the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of 98.9% on GPT-4o (29 May 2025 release) and 99.8% on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLM alignment mechanisms.

摘要: 对抗性越狱攻击的最新进展暴露了大型语言模型（LLM）中的关键漏洞，从而能够通过日益复杂的提示操作来规避对齐保障措施。根据我们的实验，我们发现越狱策略的有效性受到被攻击LLM理解能力的影响。基于这一见解，我们提出了一个功能感知的多加密框架（MEF），用于评估黑匣子LLM中的漏洞。具体来说，MEF首先对LLM的理解能力水平进行分类，然后相应地应用不同的策略：对于理解能力有限的模型，MEF采用Du + En 1策略，将分层语义突变与加密技术集成在一起，更有效地帮助规避LLM在输入和推理阶段的防御。对于理解能力强的模型，MEF使用更复杂的Fu+ En 1 + En 2策略，其中额外的双端加密技术应用于LLM的响应，进一步有助于在输出阶段逃避LLM的防御。实验结果证明了我们方法的有效性，在GPT-4 o（2025年5月29日发布）和GPT-4.1（2025年7月8日发布）上的攻击成功率分别为98.9%和99.8%。我们的工作有助于更深入地了解当前LLM对齐机制中的漏洞。



## **20. Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks**

使用LLM的显式漏洞生成：对抗性攻击之外的调查 cs.SE

Accepted to ICSME 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.10054v2) [paper-pdf](http://arxiv.org/pdf/2507.10054v2)

**Authors**: Emir Bosnak, Sahand Moslemi, Mayasah Lami, Anil Koyuncu

**Abstract**: Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities, this study examines a more direct threat: open-source LLMs generating vulnerable code when prompted. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and prompt phrasing across structured templates; and (2) Reverse Prompting, which derives natural-language prompts from real vulnerable code samples. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, Gemma) using static analysis to assess both the presence and correctness of generated vulnerabilities. Our results show that all models frequently generate the requested vulnerabilities, though with significant performance differences. Gemma achieves the highest correctness for memory vulnerabilities under Dynamic Prompting (e.g., 98.6% for buffer overflows), while Qwen2 demonstrates the most balanced performance across all tasks. We find that professional personas (e.g., "DevOps Engineer") consistently elicit higher success rates than student personas, and that the effectiveness of direct versus indirect phrasing is inverted depending on the prompting strategy. Vulnerability reproduction accuracy follows a non-linear pattern with code complexity, peaking in a moderate range. Our findings expose how LLMs' reliance on pattern recall over semantic reasoning creates significant blind spots in their safety alignments, particularly for requests framed as plausible professional tasks.

摘要: 大型语言模型（LLM）越来越多地被用作代码助手，但当被明确要求生成不安全代码时，它们的行为仍然知之甚少。虽然之前的研究重点是无意的漏洞，但这项研究考察了一个更直接的威胁：开源LLM在提示时生成易受攻击的代码。我们提出了一种双重实验设计：（1）动态预算处理，它系统地改变结构化模板中的漏洞类型、用户角色和提示措辞;（2）反向预算处理，它从真正的脆弱代码样本中派生自然语言提示。我们使用静态分析评估了三个开源7 B参数模型（Qwen 2、Mistral、Gemma），以评估生成漏洞的存在性和正确性。我们的结果表明，所有模型都会经常生成请求的漏洞，尽管性能差异很大。Gemma在动态预算分配下实现了最高的内存漏洞正确性（例如，缓冲区溢出为98.6%），而Qwen 2在所有任务中表现出最平衡的性能。我们发现专业角色（例如，“DevOps工程师”）始终比学生角色获得更高的成功率，并且直接措辞与间接措辞的有效性根据提示策略而颠倒。漏洞复制准确性遵循具有代码复杂性的非线性模式，峰值在中等范围内。我们的研究结果揭示了LLM对模式回忆而不是语义推理的依赖如何在其安全性排列中造成了显着的盲点，特别是对于被定义为看似合理的专业任务的请求。



## **21. Optimizing Privacy-Utility Trade-off in Decentralized Learning with Generalized Correlated Noise**

广义相关噪声下分散学习中隐私-效用权衡的优化 cs.LG

6 pages, 5 figures, accepted at IEEE ITW 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2501.14644v2) [paper-pdf](http://arxiv.org/pdf/2501.14644v2)

**Authors**: Angelo Rodio, Zheng Chen, Erik G. Larsson

**Abstract**: Decentralized learning enables distributed agents to collaboratively train a shared machine learning model without a central server, through local computation and peer-to-peer communication. Although each agent retains its dataset locally, sharing local models can still expose private information about the local training datasets to adversaries. To mitigate privacy attacks, a common strategy is to inject random artificial noise at each agent before exchanging local models between neighbors. However, this often leads to utility degradation due to the negative effects of cumulated artificial noise on the learning algorithm. In this work, we introduce CorN-DSGD, a novel covariance-based framework for generating correlated privacy noise across agents, which unifies several state-of-the-art methods as special cases. By leveraging network topology and mixing weights, CorN-DSGD optimizes the noise covariance to achieve network-wide noise cancellation. Experimental results show that CorN-DSGD cancels more noise than existing pairwise correlation schemes, improving model performance under formal privacy guarantees.

摘要: 去中心化学习使分布式代理能够通过本地计算和点对点通信，在没有中央服务器的情况下协作训练共享机器学习模型。尽管每个代理都在本地保留其数据集，但共享本地模型仍然可能向对手暴露有关本地训练数据集的私人信息。为了减轻隐私攻击，一种常见的策略是在邻居之间交换本地模型之前向每个代理注入随机人工噪音。然而，由于累积人工噪音对学习算法的负面影响，这通常会导致效用下降。在这项工作中，我们引入了CorN-DBCD，这是一种新型的基于协方差的框架，用于在代理之间生成相关隐私噪音，它将多种最先进的方法统一为特例。通过利用网络布局和混合权重，CorN-DBCD优化噪音协方差，以实现网络范围的噪音消除。实验结果表明，CorN-DBCD比现有的成对相关方案消除了更多的噪音，在形式隐私保证下提高了模型性能。



## **22. Restricted Boltzmann machine as a probabilistic Enigma**

作为概率谜的限制Boltzmann机 cond-mat.stat-mech

7 pages, 4 figures

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17236v1) [paper-pdf](http://arxiv.org/pdf/2507.17236v1)

**Authors**: Bin Chen, Weichao Yu

**Abstract**: We theoretically propose a symmetric encryption scheme based on Restricted Boltzmann Machines that functions as a probabilistic Enigma device, encoding information in the marginal distributions of visible states while utilizing bias permutations as cryptographic keys. Theoretical analysis reveals significant advantages including factorial key space growth through permutation matrices, excellent diffusion properties, and computational complexity rooted in sharp P-complete problems that resist quantum attacks. Compatible with emerging probabilistic computing hardware, the scheme establishes an asymmetric computational barrier where legitimate users decrypt efficiently while adversaries face exponential costs. This framework unlocks probabilistic computers' potential for cryptographic systems, offering an emerging encryption paradigm between classical and quantum regimes for post-quantum security.

摘要: 从理论上讲，我们提出了一种基于受限制的Boltzmann机的对称加密方案，该方案充当概率谜设备，以可见状态的边缘分布对信息进行编码，同时利用偏差排列作为密钥。理论分析揭示了显着的优势，包括通过置换矩阵的因乘密钥空间增长、出色的扩散性质以及源于抵抗量子攻击的尖锐P完全问题的计算复杂性。该方案与新兴的概率计算硬件兼容，建立了一个不对称的计算障碍，合法用户可以有效解密，而对手则面临指数级成本。该框架释放了概率计算机在密码系统中的潜力，为后量子安全提供了经典和量子机制之间的新兴加密范式。



## **23. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

Gungnir：利用图像中的风格特征对扩散模型进行后门攻击 cs.CV

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2502.20650v4) [paper-pdf](http://arxiv.org/pdf/2502.20650v4)

**Authors**: Yu Pan, Jiahao Chen, Bingrong Dai, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.

摘要: 近年来，扩散模型（DM）在图像生成领域取得了重大进展。然而，根据当前的研究，DM很容易受到后门攻击，后门攻击允许攻击者通过输入包含隐蔽触发器（例如特定的视觉补丁或短语）的数据来控制模型的输出。现有的防御策略完全可以通过后门检测和触发器倒置来阻止此类攻击，因为以前的攻击方法受到有限的输入空间和低维触发器的限制。例如，视觉触发器很容易被防御者观察到，基于文本或基于注意力的触发器更容易受到神经网络检测的影响。为了探索DM中后门攻击的更多可能性，我们提出了Gungnir，这是一种新颖的方法，使攻击者能够通过输入图像中的风格触发器激活DM中的后门。我们的方法首次提出使用风格特征作为触发器，并通过引入重建对抗噪音（RAN）和短期时间间隔保留（STTR）在图像到图像任务中成功实施后门攻击。我们的技术生成的嵌入式图像在感知上与干净图像无法区分，从而绕过了手动检查和自动检测神经网络。实验表明，贡尼尔可以轻松绕过现有的防御方法。在现有的DM防御框架中，我们的方法实现了0后门检测率（BDR）。我们的代码可在https://github.com/paoche11/Gungnir上获得。



## **24. Advancing Robustness in Deep Reinforcement Learning with an Ensemble Defense Approach**

通过集群防御方法提高深度强化学习的鲁棒性 cs.LG

6 pages, 4 figures, 2 tables

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.17070v1) [paper-pdf](http://arxiv.org/pdf/2507.17070v1)

**Authors**: Adithya Mohan, Dominik Rößle, Daniel Cremers, Torsten Schön

**Abstract**: Recent advancements in Deep Reinforcement Learning (DRL) have demonstrated its applicability across various domains, including robotics, healthcare, energy optimization, and autonomous driving. However, a critical question remains: How robust are DRL models when exposed to adversarial attacks? While existing defense mechanisms such as adversarial training and distillation enhance the resilience of DRL models, there remains a significant research gap regarding the integration of multiple defenses in autonomous driving scenarios specifically. This paper addresses this gap by proposing a novel ensemble-based defense architecture to mitigate adversarial attacks in autonomous driving. Our evaluation demonstrates that the proposed architecture significantly enhances the robustness of DRL models. Compared to the baseline under FGSM attacks, our ensemble method improves the mean reward from 5.87 to 18.38 (over 213% increase) and reduces the mean collision rate from 0.50 to 0.09 (an 82% decrease) in the highway scenario and merge scenario, outperforming all standalone defense strategies.

摘要: 深度强化学习（DRL）的最新进展已经证明了其在各个领域的适用性，包括机器人、医疗保健、能源优化和自动驾驶。然而，一个关键问题仍然存在：DRL模型在遭受对抗攻击时有多稳健？虽然对抗训练和提炼等现有防御机制增强了DRL模型的弹性，但关于自动驾驶场景中多种防御的集成仍然存在显着的研究空白。本文通过提出一种新颖的基于集成的防御架构来缓解自动驾驶中的对抗攻击来解决这一差距。我们的评估表明，提出的架构显着增强了DRL模型的鲁棒性。与FGSM攻击下的基线相比，我们的集成方法在高速公路场景和合并场景中将平均奖励从5.87提高到18.38（增加超过213%），并将平均碰撞率从0.50降低到0.09（减少82%），优于所有独立防御策略。



## **25. When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs**

当LLM复制到思考时：揭示推理LLM中的复制引导攻击 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16773v1) [paper-pdf](http://arxiv.org/pdf/2507.16773v1)

**Authors**: Yue Li, Xiao Li, Hao Wu, Yue Zhang, Fengyuan Xu, Xiuzhen Cheng, Sheng Zhong

**Abstract**: Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.

摘要: 大型语言模型（LLM）已成为自动代码分析的组成部分，可以实现漏洞检测和代码理解等任务。然而，它们的集成引入了新颖的攻击表面。在本文中，我们识别并研究了一类新的基于预算的攻击，称为复制引导攻击（CGA），它利用了具有推理能力的LLM固有的复制倾向。通过将精心制作的触发器注入外部代码片段中，对手可以诱导模型在推理期间复制恶意内容。这种行为导致两类漏洞：推理长度操纵，其中模型生成异常短或过长的推理痕迹;和推理结果操纵，其中模型生成误导性或不正确的结论。我们将CGA形式化为一个优化问题，并提出一种基于梯度的方法来合成有效的触发器。对最先进推理LLM的实证评估表明，CGA可靠地在代码分析任务中引发无限循环、提前终止、错误拒绝和语义扭曲。虽然在目标环境中非常有效，但由于计算限制，我们观察到在不同提示中推广CGA存在挑战，这为未来的研究提出了一个悬而未决的问题。我们的研究结果揭示了LLM驱动的开发管道中一个关键但未充分探索的漏洞，并呼吁在预算级防御机制方面紧急取得进展。



## **26. ShadowCode: Towards (Automatic) External Prompt Injection Attack against Code LLMs**

ShadowCode：针对代码LLM的（自动）外部提示注入攻击 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2407.09164v6) [paper-pdf](http://arxiv.org/pdf/2407.09164v6)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Bingrun Yang, Yiling He, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: Recent advancements have led to the widespread adoption of code-oriented large language models (Code LLMs) for programming tasks. Despite their success in deployment, their security research is left far behind. This paper introduces a new attack paradigm: (automatic) external prompt injection against Code LLMs, where attackers generate concise, non-functional induced perturbations and inject them within a victim's code context. These induced perturbations can be disseminated through commonly used dependencies (e.g., packages or RAG's knowledge base), manipulating Code LLMs to achieve malicious objectives during the code completion process. Compared to existing attacks, this method is more realistic and threatening: it does not necessitate control over the model's training process, unlike backdoor attacks, and can achieve specific malicious objectives that are challenging for adversarial attacks. Furthermore, we propose ShadowCode, a simple yet effective method that automatically generates induced perturbations based on code simulation to achieve effective and stealthy external prompt injection. ShadowCode designs its perturbation optimization objectives by simulating realistic code contexts and employs a greedy optimization approach with two enhancement modules: forward reasoning enhancement and keyword-based perturbation design. We evaluate our method across 13 distinct malicious objectives, generating 31 threat cases spanning three popular programming languages. Our results demonstrate that ShadowCode successfully attacks three representative open-source Code LLMs (achieving up to a 97.9% attack success rate) and two mainstream commercial Code LLM-integrated applications (with over 90% attack success rate) across all threat cases, using only a 12-token non-functional induced perturbation. The code is available at https://github.com/LianPing-cyber/ShadowCodeEPI.

摘要: 最近的进步导致面向代码的大型语言模型（Code LLM）被广泛采用来进行编程任务。尽管他们在部署方面取得了成功，但他们的安全研究却远远落后。本文引入了一种新的攻击范式：针对Code LLM的（自动）外部提示注入，攻击者生成简洁的、非功能性的诱导扰动，并将其注入到受害者的代码上下文中。这些引发的扰动可以通过常用的依赖性传播（例如，包或RAG的知识库），在代码完成过程中操纵代码LLM以实现恶意目标。与现有的攻击相比，这种方法更现实、更具威胁性：与后门攻击不同，它不需要控制模型的训练过程，并且可以实现对对抗性攻击具有挑战性的特定恶意目标。此外，我们提出了ShadowCode，这是一种简单而有效的方法，可以根据代码模拟自动生成诱导扰动，以实现有效且隐蔽的外部提示注入。ShadowCode通过模拟现实代码上下文来设计其扰动优化目标，并采用具有两个增强模块的贪婪优化方法：前向推理增强和基于关键字的扰动设计。我们针对13个不同的恶意目标评估我们的方法，生成了跨越三种流行编程语言的31个威胁案例。我们的结果表明，ShadowCode仅使用12个令牌的非功能性诱导扰动，就成功攻击了所有威胁案例中的三个代表性开源Code LLM（攻击成功率高达97.9%）和两个主流商业Code LLM集成应用程序（攻击成功率超过90%）。该代码可在https://github.com/LianPing-cyber/ShadowCodeEPI上获取。



## **27. The Cost of Compression: Tight Quadratic Black-Box Attacks on Sketches for $\ell_2$ Norm Estimation**

压缩的成本：对草图进行严格的二次黑匣子攻击，以获取$\ell_2 $ Norm估计 cs.LG

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16345v1) [paper-pdf](http://arxiv.org/pdf/2507.16345v1)

**Authors**: Sara Ahmadian, Edith Cohen, Uri Stemmer

**Abstract**: Dimensionality reduction via linear sketching is a powerful and widely used technique, but it is known to be vulnerable to adversarial inputs. We study the black-box adversarial setting, where a fixed, hidden sketching matrix A in $R^{k X n}$ maps high-dimensional vectors v $\in R^n$ to lower-dimensional sketches A v in $R^k$, and an adversary can query the system to obtain approximate ell2-norm estimates that are computed from the sketch.   We present a universal, nonadaptive attack that, using tilde(O)($k^2$) queries, either causes a failure in norm estimation or constructs an adversarial input on which the optimal estimator for the query distribution (used by the attack) fails. The attack is completely agnostic to the sketching matrix and to the estimator: It applies to any linear sketch and any query responder, including those that are randomized, adaptive, or tailored to the query distribution.   Our lower bound construction tightly matches the known upper bounds of tilde(Omega)($k^2$), achieved by specialized estimators for Johnson Lindenstrauss transforms and AMS sketches. Beyond sketching, our results uncover structural parallels to adversarial attacks in image classification, highlighting fundamental vulnerabilities of compressed representations.

摘要: 通过线性草图减少维度是一种强大且广泛使用的技术，但众所周知，它容易受到对抗输入的影响。我们研究黑匣子对抗设置，其中$R #{k X n}中的固定、隐藏的草图矩阵A将R ##中的多维载体v $\映射到$R #中的低维草图A v #中的低维草图A v，并且对手可以查询系统以获得从草图计算的大约ell 2-模估计。   我们提出了一种通用的、非适应性攻击，它使用波浪形（O）（$k ' 2 $）查询，要么导致规范估计失败，要么构建一个对抗性输入，而查询分布的最佳估计器（由攻击使用）失败。该攻击对草图矩阵和估计器完全不可知：它适用于任何线性草图和任何查询响应器，包括随机化、自适应或针对查询分布定制的那些。   我们的下限结构与波浪形（Omega）的已知上限（$k ' 2 $）紧密匹配，该上限由Johnson Lindenstrauss变换和AMS草图的专门估计器实现。除了草图之外，我们的结果还揭示了图像分类中与对抗攻击的结构相似之处，凸显了压缩表示的基本漏洞。



## **28. Talking Like a Phisher: LLM-Based Attacks on Voice Phishing Classifiers**

像网络钓鱼者一样说话：基于LLM的语音网络钓鱼分类器攻击 cs.CR

Accepted by EAI ICDF2C 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16291v1) [paper-pdf](http://arxiv.org/pdf/2507.16291v1)

**Authors**: Wenhao Li, Selvakumar Manickam, Yung-wey Chong, Shankar Karuppayah

**Abstract**: Voice phishing (vishing) remains a persistent threat in cybersecurity, exploiting human trust through persuasive speech. While machine learning (ML)-based classifiers have shown promise in detecting malicious call transcripts, they remain vulnerable to adversarial manipulations that preserve semantic content. In this study, we explore a novel attack vector where large language models (LLMs) are leveraged to generate adversarial vishing transcripts that evade detection while maintaining deceptive intent. We construct a systematic attack pipeline that employs prompt engineering and semantic obfuscation to transform real-world vishing scripts using four commercial LLMs. The generated transcripts are evaluated against multiple ML classifiers trained on a real-world Korean vishing dataset (KorCCViD) with statistical testing. Our experiments reveal that LLM-generated transcripts are both practically and statistically effective against ML-based classifiers. In particular, transcripts crafted by GPT-4o significantly reduce classifier accuracy (by up to 30.96%) while maintaining high semantic similarity, as measured by BERTScore. Moreover, these attacks are both time-efficient and cost-effective, with average generation times under 9 seconds and negligible financial cost per query. The results underscore the pressing need for more resilient vishing detection frameworks and highlight the imperative for LLM providers to enforce stronger safeguards against prompt misuse in adversarial social engineering contexts.

摘要: 语音网络钓鱼（网络钓鱼）仍然是网络安全领域的一个持续威胁，它通过有说服力的言论来利用人类的信任。虽然基于机器学习（ML）的分类器在检测恶意通话记录方面表现出了希望，但它们仍然容易受到保留语义内容的对抗性操纵的影响。在这项研究中，我们探索了一种新型的攻击载体，其中利用大型语言模型（LLM）来生成对抗性的钓鱼笔录，以逃避检测，同时保持欺骗意图。我们构建了一个系统性攻击管道，该管道采用即时工程和语义混淆来使用四个商业LLM来转换现实世界的视频脚本。针对在现实世界的韩国钓鱼数据集（KorCCCViD）上训练的多个ML分类器进行评估生成的成绩单，并进行统计测试。我们的实验表明，LLM生成的成绩单对于基于ML的分类器在实践和统计上都有效。特别是，GPT-4 o制作的文字记录显着降低了分类器的准确性（高达30.96%），同时保持了高的语义相似性（由BERTScore衡量）。此外，这些攻击兼具时间效率和成本效益，平均生成时间不到9秒，每次查询的财务成本可以忽略不计。结果强调了对更具弹性的钓鱼检测框架的迫切需要，并强调了LLM提供者必须实施更强有力的保障措施，防止在对抗性社会工程环境中迅速滥用。



## **29. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

使用具有指定概率操纵的白盒对抗攻击对DNN模型进行所有权验证 cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2505.17579v2) [paper-pdf](http://arxiv.org/pdf/2505.17579v2)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Ishobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.

摘要: 在本文中，我们提出了一种新的框架，用于图像分类任务的深度神经网络（DNN）模型的所有权验证。它允许合法所有者和第三方验证型号身份，而无需出示原始型号。我们假设一个灰盒场景，其中未经授权的用户拥有从原始模型非法复制的模型，在云环境中提供服务，用户抛出图像并接收分类结果作为输出类的概率分布。该框架应用白盒对抗攻击来将特定类的输出概率与指定值对齐。由于对原始模型的了解，它使所有者能够生成此类对抗性示例。我们通过引入控制参数，提出了一种基于迭代快速梯度符号法（FGSM）的简单但有效的对抗攻击方法。实验结果证实了使用对抗攻击识别DNN模型的有效性。



## **30. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

拉开帷幕：通过对比辅助网络的无监督对抗检测 cs.CV

Accepted at SafeMM-AI @ ICCV 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2502.09110v2) [paper-pdf](http://arxiv.org/pdf/2502.09110v2)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.

摘要: 深度学习模型广泛应用于安全关键应用中，但仍然容易受到对抗攻击--难以察觉的扰动，会显着降低模型性能。传统的防御机制主要专注于增强模型稳健性或独立检测对抗输入。在这项工作中，我们提出了一种通过对比辅助网络（U-CAN）的无监督对抗检测，以发现辅助特征表示中的对抗行为，而不需要对抗示例。U-CAN嵌入在目标模型的选定中间层中。这些辅助网络由投影层和基于ArcFace的线性层组成，可以细化特征表示，以更有效地区分良性输入和对抗输入。跨多个数据集（CIFAR-10、Mammals和ImageNet的一个子集）和架构（ResNet-50、VGG-16和ViT）的综合实验表明，我们的方法超越了现有的无监督对抗检测技术，在针对四种不同的攻击方法的情况下获得了优异的F1分数。提出的框架为增强深度学习系统的安全性和可靠性提供了可扩展且有效的解决方案。



## **31. Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models**

优质文本，稳健的视觉：语言在增强视觉语言模型视觉稳健性方面的作用 cs.CV

ACMMM 2025 Accepted

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16257v1) [paper-pdf](http://arxiv.org/pdf/2507.16257v1)

**Authors**: Futa Waseda, Saku Sugawara, Isao Echizen

**Abstract**: Defending pre-trained vision-language models (VLMs), such as CLIP, against adversarial attacks is crucial, as these models are widely used in diverse zero-shot tasks, including image classification. However, existing adversarial training (AT) methods for robust fine-tuning largely overlook the role of language in enhancing visual robustness. Specifically, (1) supervised AT methods rely on short texts (e.g., class labels) to generate adversarial perturbations, leading to overfitting to object classes in the training data, and (2) unsupervised AT avoids this overfitting but remains suboptimal against practical text-guided adversarial attacks due to its lack of semantic guidance. To address these limitations, we propose Quality Text-guided Adversarial Fine-Tuning (QT-AFT), which leverages high-quality captions during training to guide adversarial examples away from diverse semantics present in images. This enables the visual encoder to robustly recognize a broader range of image features even under adversarial noise, thereby enhancing robustness across diverse downstream tasks. QT-AFT overcomes the key weaknesses of prior methods -- overfitting in supervised AT and lack of semantic awareness in unsupervised AT -- achieving state-of-the-art zero-shot adversarial robustness and clean accuracy, evaluated across 16 zero-shot datasets. Furthermore, our comprehensive study uncovers several key insights into the role of language in enhancing vision robustness; for example, describing object properties in addition to object names further enhances zero-shot robustness. Our findings point to an urgent direction for future work -- centering high-quality linguistic supervision in robust visual representation learning.

摘要: 保护预先训练的视觉语言模型（VLM）（例如CLIP）免受对抗攻击至关重要，因为这些模型广泛用于各种零射击任务，包括图像分类。然而，现有的用于鲁棒微调的对抗训练（AT）方法在很大程度上忽视了语言在增强视觉鲁棒性方面的作用。具体来说，（1）监督AT方法依赖于短文本（例如，类标签）来生成对抗性扰动，导致对训练数据中的对象类的过度匹配，以及（2）无监督AT避免了这种过度匹配，但由于缺乏语义指导，对于实际的文本引导对抗性攻击仍然次优。为了解决这些限制，我们提出了优质文本引导的对抗微调（QT-AFT），它利用训练期间的高质量字幕来引导对抗示例远离图像中存在的各种语义。这使得视觉编码器即使在对抗性噪音下也能够稳健地识别更广泛的图像特征，从而增强各种下游任务的稳健性。QT-AFT克服了现有方法的关键弱点--监督AT中的过度匹配和无监督AT中缺乏语义意识--实现了最先进的零射击对抗鲁棒性和清晰的准确性，并在16个零射击数据集中进行了评估。此外，我们的全面研究揭示了语言在增强视觉鲁棒性方面作用的几个关键见解;例如，除了对象名称之外，描述对象属性进一步增强了零镜头鲁棒性。我们的研究结果指出了未来工作的紧迫方向--以稳健的视觉表示学习为中心的高质量语言监督。



## **32. Pulse-Level Simulation of Crosstalk Attacks on Superconducting Quantum Hardware**

超导体量子硬件串话攻击的脉冲级模拟 quant-ph

This paper has been accepted to the Security, Privacy, and Resilience  Workshop at IEEE Quantum Week (QCE 2025) and will appear in the workshop  proceedings

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16181v1) [paper-pdf](http://arxiv.org/pdf/2507.16181v1)

**Authors**: Syed Emad Uddin Shubha, Tasnuva Farheen

**Abstract**: Hardware crosstalk in multi-tenant superconducting quantum computers poses a severe security threat, allowing adversaries to induce targeted errors across tenant boundaries by injecting carefully engineered pulses. We present a simulation-based study of active crosstalk attacks at the pulse level, analyzing how adversarial control of pulse timing, shape, amplitude, and coupling can disrupt a victim's computation. Our framework models the time-dependent dynamics of a three-qubit system in the rotating frame, capturing both always-on couplings and injected drive pulses. We examine two attack strategies: attacker-first (pulse before victim operation) and victim-first (pulse after), and systematically identify the pulse and coupling configurations that cause the largest logical errors. Protocol-level experiments on quantum coin flip and XOR classification circuits show that some protocols are highly vulnerable to these attacks, while others remain robust. Based on these findings, we discuss practical methods for detection and mitigation to improve security in quantum cloud platforms.

摘要: 多租户高温量子计算机中的硬件串烧构成了严重的安全威胁，使对手能够通过注入精心设计的脉冲来跨越租户边界引发有针对性的错误。我们对脉冲级的主动串话攻击进行了一项基于模拟的研究，分析脉冲定时、形状、幅度和耦合的对抗控制如何扰乱受害者的计算。我们的框架对旋转框架中三量子位系统的时间相关动态进行建模，捕获始终在线的耦合和注入的驱动脉冲。我们研究了两种攻击策略：攻击者优先（受害者操作前脉冲）和受害者优先（受害者操作后脉冲），并系统性地识别导致最大逻辑错误的脉冲和耦合配置。在量子硬币翻转和XOR分类电路上进行的协议级实验表明，一些协议非常容易受到这些攻击，而另一些协议仍然很健壮。基于这些发现，我们讨论了检测和缓解的实用方法，以提高量子云平台的安全性。



## **33. Attacking interpretable NLP systems**

攻击可解释的NLP系统 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16164v1) [paper-pdf](http://arxiv.org/pdf/2507.16164v1)

**Authors**: Eldor Abdukhamidov, Tamer Abuhmed, Joanna C. S. Santos, Mohammed Abuhamad

**Abstract**: Studies have shown that machine learning systems are vulnerable to adversarial examples in theory and practice. Where previous attacks have focused mainly on visual models that exploit the difference between human and machine perception, text-based models have also fallen victim to these attacks. However, these attacks often fail to maintain the semantic meaning of the text and similarity. This paper introduces AdvChar, a black-box attack on Interpretable Natural Language Processing Systems, designed to mislead the classifier while keeping the interpretation similar to benign inputs, thus exploiting trust in system transparency. AdvChar achieves this by making less noticeable modifications to text input, forcing the deep learning classifier to make incorrect predictions and preserve the original interpretation. We use an interpretation-focused scoring approach to determine the most critical tokens that, when changed, can cause the classifier to misclassify the input. We apply simple character-level modifications to measure the importance of tokens, minimizing the difference between the original and new text while generating adversarial interpretations similar to benign ones. We thoroughly evaluated AdvChar by testing it against seven NLP models and three interpretation models using benchmark datasets for the classification task. Our experiments show that AdvChar can significantly reduce the prediction accuracy of current deep learning models by altering just two characters on average in input samples.

摘要: 研究表明，机器学习系统在理论和实践中容易受到对抗性例子的影响。之前的攻击主要集中在利用人类和机器感知之间差异的视觉模型上，而基于文本的模型也成为这些攻击的受害者。然而，这些攻击往往无法保持文本的语义含义和相似性。本文介绍了AdvChar，这是一种针对可解释自然语言处理系统的黑匣子攻击，旨在误导分类器，同时保持解释类似于良性输入，从而利用对系统透明度的信任。AdvChar通过对文本输入进行不太明显的修改来实现这一目标，迫使深度学习分类器做出错误的预测并保留原始解释。我们使用以解释为中心的评分方法来确定最关键的标记，这些标记在更改时可能会导致分类器对输入进行错误分类。我们应用简单的字符级修改来衡量代币的重要性，最大限度地减少原始文本和新文本之间的差异，同时生成类似于良性解释的对抗性解释。我们通过使用分类任务的基准数据集针对七个NLP模型和三个解释模型进行测试，彻底评估了AdvChar。我们的实验表明，AdvChar只需平均改变输入样本中的两个字符即可显着降低当前深度学习模型的预测准确性。



## **34. DP2Guard: A Lightweight and Byzantine-Robust Privacy-Preserving Federated Learning Scheme for Industrial IoT**

DP 2Guard：一种用于工业物联网的轻量级、拜占庭鲁棒的隐私保护联邦学习计划 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16134v1) [paper-pdf](http://arxiv.org/pdf/2507.16134v1)

**Authors**: Baofu Han, Bing Li, Yining Qi, Raja Jurdak, Kaibin Huang, Chau Yuen

**Abstract**: Privacy-Preserving Federated Learning (PPFL) has emerged as a secure distributed Machine Learning (ML) paradigm that aggregates locally trained gradients without exposing raw data. To defend against model poisoning threats, several robustness-enhanced PPFL schemes have been proposed by integrating anomaly detection. Nevertheless, they still face two major challenges: (1) the reliance on heavyweight encryption techniques results in substantial communication and computation overhead; and (2) single-strategy defense mechanisms often fail to provide sufficient robustness against adaptive adversaries. To overcome these challenges, we propose DP2Guard, a lightweight PPFL framework that enhances both privacy and robustness. DP2Guard leverages a lightweight gradient masking mechanism to replace costly cryptographic operations while ensuring the privacy of local gradients. A hybrid defense strategy is proposed, which extracts gradient features using singular value decomposition and cosine similarity, and applies a clustering algorithm to effectively identify malicious gradients. Additionally, DP2Guard adopts a trust score-based adaptive aggregation scheme that adjusts client weights according to historical behavior, while blockchain records aggregated results and trust scores to ensure tamper-proof and auditable training. Extensive experiments conducted on two public datasets demonstrate that DP2Guard effectively defends against four advanced poisoning attacks while ensuring privacy with reduced communication and computation costs.

摘要: 隐私保护联邦学习（PPFL）已成为一种安全的分布式机器学习（ML）范式，可以在不暴露原始数据的情况下聚合本地训练的梯度。为了抵御模型中毒威胁，通过集成异常检测提出了几种鲁棒性增强的PPFL方案。尽管如此，它们仍然面临两个主要挑战：（1）对重量级加密技术的依赖导致大量通信和计算负担;（2）单策略防御机制通常无法提供足够的鲁棒性来对抗自适应对手。为了克服这些挑战，我们提出了DP 2Guard，这是一个轻量级PPFL框架，可以增强隐私性和稳健性。DP 2Guard利用轻量级的梯度屏蔽机制来取代昂贵的加密操作，同时确保本地梯度的隐私。提出了一种混合防御策略，利用奇异值分解和cos相似度提取梯度特征，并应用集群算法有效识别恶意梯度。此外，DP 2Guard采用基于信任分数的自适应聚合方案，根据历史行为调整客户权重，而区块链记录聚合结果和信任分数，以确保防篡改和可审计的训练。在两个公共数据集上进行的大量实验表明，DP 2Guard可以有效防御四种高级中毒攻击，同时确保隐私并降低通信和计算成本。



## **35. DP-TLDM: Differentially Private Tabular Latent Diffusion Model**

DP-TLDM：差异私人表格潜在扩散模型 cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2403.07842v2) [paper-pdf](http://arxiv.org/pdf/2403.07842v2)

**Authors**: Chaoyi Zhu, Jiayi Tang, Juan F. Pérez, Marten van Dijk, Lydia Y. Chen

**Abstract**: Synthetic data from generative models emerges as the privacy-preserving data sharing solution. Such a synthetic data set shall resemble the original data without revealing identifiable private information. Till date, the prior focus on limited types of tabular synthesizers and a small number of privacy attacks, particularly on Generative Adversarial Networks, and overlooks membership inference attacks and defense strategies, i.e., differential privacy. Motivated by the conundrum of keeping high data quality and low privacy risk of synthetic data tables, we propose DPTLDM, Differentially Private Tabular Latent Diffusion Model, which is composed of an autoencoder network to encode the tabular data and a latent diffusion model to synthesize the latent tables. Following the emerging f-DP framework, we apply DP-SGD to train the auto-encoder in combination with batch clipping and use the separation value as the privacy metric to better capture the privacy gain from DP algorithms. Our empirical evaluation demonstrates that DPTLDM is capable of achieving a meaningful theoretical privacy guarantee while also significantly enhancing the utility of synthetic data. Specifically, compared to other DP-protected tabular generative models, DPTLDM improves the synthetic quality by an average of 35% in data resemblance, 15% in the utility for downstream tasks, and 50% in data discriminability, all while preserving a comparable level of privacy risk.

摘要: 来自生成模型的合成数据成为保护隐私的数据共享解决方案。此类合成数据集应与原始数据相似，而不会透露可识别的私人信息。迄今为止，之前的重点是有限类型的表格合成器和少数隐私攻击，特别是生成式对抗网络，并忽视了成员资格推断攻击和防御策略，即差异隐私。出于保持合成数据表的高数据质量和低隐私风险的难题的动机，我们提出了DPTLDM，即差异私有表格潜在扩散模型，它由一个用于编码表格数据的自动编码器网络和一个用于合成潜在表的潜在扩散模型组成。遵循新兴的f-DP框架，我们应用DP-BCD结合批量剪裁来训练自动编码器，并使用分离值作为隐私指标，以更好地捕捉DP算法的隐私收益。我们的实证评估表明，DPTLDM能够实现有意义的理论隐私保证，同时还显着提高合成数据的实用性。具体来说，与其他DP保护的表格生成模型相比，DPTLDM将合成质量在数据相似性方面平均提高了35%，下游任务的实用性提高了15%，数据区分性提高了50%，同时保留了相当水平的隐私风险。



## **36. Erasing Conceptual Knowledge from Language Models**

从语言模型中删除概念知识 cs.CL

Project Page: https://elm.baulab.info

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2410.02760v3) [paper-pdf](http://arxiv.org/pdf/2410.02760v3)

**Authors**: Rohit Gandikota, Sheridan Feucht, Samuel Marks, David Bau

**Abstract**: In this work, we introduce Erasure of Language Memory (ELM), a principled approach to concept-level unlearning that operates by matching distributions defined by the model's own introspective classification capabilities. Our key insight is that effective unlearning should leverage the model's ability to evaluate its own knowledge, using the language model itself as a classifier to identify and reduce the likelihood of generating content related to undesired concepts. ELM applies this framework to create targeted low-rank updates that reduce generation probabilities for concept-specific content while preserving the model's broader capabilities. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative evaluation reveals that ELM-modified models achieve near-random performance on assessments targeting erased concepts, while simultaneously preserving generation coherence, maintaining benchmark performance on unrelated tasks, and exhibiting strong robustness to adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info

摘要: 在这项工作中，我们引入了语言记忆擦除（ELM），这是一种概念级去学习的原则方法，通过匹配模型自己的内省分类能力定义的分布来操作。我们的主要见解是，有效的取消学习应该利用模型评估其自身知识的能力，使用语言模型本身作为分类器来识别和降低生成与不需要的概念相关的内容的可能性。ELM应用此框架来创建有针对性的低等级更新，以降低特定概念内容的生成概率，同时保留模型更广泛的功能。我们展示了ELM在生物安全、网络安全和文学领域擦除任务方面的功效。比较评估表明，ELM修改后的模型在针对已删除概念的评估中实现了近乎随机的性能，同时保持世代一致性，在不相关任务上保持基准性能，并对对抗性攻击表现出强大的鲁棒性。我们的代码、数据和训练模型可在https://elm.baulab.info上获取



## **37. Disrupting Semantic and Abstract Features for Better Adversarial Transferability**

破坏语义和抽象特征以提高对抗性可转移性 cs.CV

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.16052v1) [paper-pdf](http://arxiv.org/pdf/2507.16052v1)

**Authors**: Yuyang Luo, Xiaosen Wang, Zhijin Ge, Yingzhe He

**Abstract**: Adversarial examples pose significant threats to deep neural networks (DNNs), and their property of transferability in the black-box setting has led to the emergence of transfer-based attacks, making it feasible to target real-world applications employing DNNs. Among them, feature-level attacks, where intermediate features are perturbed based on feature importance weight matrix computed from transformed images, have gained popularity. In this work, we find that existing feature-level attacks primarily manipulate the semantic information to derive the weight matrix. Inspired by several works that find CNNs tend to focus more on high-frequency components (a.k.a. abstract features, e.g., texture, edge, etc.), we validate that transforming images in the high-frequency space also improves transferability. Based on this finding, we propose a balanced approach called Semantic and Abstract FEatures disRuption (SAFER). Specifically, SAFER conducts BLOCKMIX on the input image and SELF-MIX on the frequency spectrum when computing the weight matrix to highlight crucial features. By using such a weight matrix, we can direct the attacker to disrupt both semantic and abstract features, leading to improved transferability. Extensive experiments on the ImageNet dataset also demonstrate the effectiveness of our method in boosting adversarial transferability.

摘要: 对抗性示例对深度神经网络（DNN）构成了重大威胁，其在黑匣子环境中的可移植性导致了基于传输的攻击的出现，使得使用DNN针对现实世界的应用程序成为可能。其中，特征级攻击越来越流行，其中根据从变换图像计算出的特征重要性权重矩阵来扰乱中间特征。在这项工作中，我们发现现有的特征级攻击主要操纵语义信息来推导权重矩阵。受几项作品的启发，这些作品发现CNN倾向于更多地关注高频分量（又名抽象特征，例如，纹理、边缘等），我们验证了在高频空间中变换图像也可以提高可移植性。基于这一发现，我们提出了一种平衡的方法，称为语义和抽象特征分解（SAGER）。具体来说，SAGER在计算权重矩阵以突出关键特征时对输入图像进行BLOCKMIX，并对频谱进行SELF-MIX。通过使用这样的权重矩阵，我们可以指示攻击者破坏语义和抽象特征，从而提高可移植性。ImageNet数据集上的大量实验也证明了我们的方法在提高对抗可移植性方面的有效性。



## **38. Does More Inference-Time Compute Really Help Robustness?**

更多的推理时间计算真的有助于鲁棒性吗？ cs.AI

Preprint

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15974v1) [paper-pdf](http://arxiv.org/pdf/2507.15974v1)

**Authors**: Tong Wu, Chong Xiang, Jiachen T. Wang, Weichen Yu, Chawin Sitawarin, Vikash Sehwag, Prateek Mittal

**Abstract**: Recently, Zaremba et al. demonstrated that increasing inference-time computation improves robustness in large proprietary reasoning LLMs. In this paper, we first show that smaller-scale, open-source models (e.g., DeepSeek R1, Qwen3, Phi-reasoning) can also benefit from inference-time scaling using a simple budget forcing strategy. More importantly, we reveal and critically examine an implicit assumption in prior work: intermediate reasoning steps are hidden from adversaries. By relaxing this assumption, we identify an important security risk, intuitively motivated and empirically verified as an inverse scaling law: if intermediate reasoning steps become explicitly accessible, increased inference-time computation consistently reduces model robustness. Finally, we discuss practical scenarios where models with hidden reasoning chains are still vulnerable to attacks, such as models with tool-integrated reasoning and advanced reasoning extraction attacks. Our findings collectively demonstrate that the robustness benefits of inference-time scaling depend heavily on the adversarial setting and deployment context. We urge practitioners to carefully weigh these subtle trade-offs before applying inference-time scaling in security-sensitive, real-world applications.

摘要: 最近，Zaremba等人证明，增加推理时计算可以提高大型专有推理LLM的鲁棒性。在本文中，我们首先展示了较小规模的开源模型（例如，DeepSeek R1、Qwen 3、Phi-reason）还可以受益于使用简单的预算强制策略的推理时扩展。更重要的是，我们揭示并批判性地检查了之前工作中的一个隐含假设：中间推理步骤对对手是隐藏的。通过放松这一假设，我们确定了一个重要的安全风险，直观的动机和经验验证的逆尺度律：如果中间的推理步骤变得显式访问，增加推理时间计算一致降低模型的鲁棒性。最后，我们讨论了具有隐藏推理链的模型仍然容易受到攻击的实际场景，例如具有工具集成推理和高级推理提取攻击的模型。我们的研究结果共同表明，推理时间缩放的鲁棒性优势在很大程度上取决于对抗性设置和部署上下文。我们敦促从业者在安全敏感的实际应用中应用推理时间缩放之前仔细权衡这些微妙的权衡。



## **39. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges**

沼泽上的对冲基金：分析区块链桥梁中的模式、漏洞和防御措施 cs.ET

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.06156v3) [paper-pdf](http://arxiv.org/pdf/2507.06156v3)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.

摘要: 区块链桥梁已成为实现不同区块链网络互操作性的重要基础设施，每月桥梁交易量超过240亿美元。然而，随着它们的日益普及，安全漏洞也不成比例地增加，使它们成为Web 3中最大的财务损失来源。为了实现跨链生态系统的稳健和可持续发展，了解和解决这些脆弱性至关重要。在这项研究中，我们对区块链桥梁设计和安全进行了全面的系统化。我们定义了三个桥梁安全先验，正式确定了13个突出桥梁的架构结构，并确定了23个基于现实世界区块链漏洞的攻击向量。在此基础上，我们评估了43种有代表性的攻击场景，并引入了一个分层的威胁模型，该模型可以捕获源链、链下和目标链组件的安全故障。   我们在静态代码和交易网络层面的分析揭示了反复出现的设计缺陷，特别是在访问控制、验证者信任假设和验证逻辑方面，并根据交易级跟踪识别了对抗行为的关键模式。为了支持未来的发展，我们提出了一个决策框架的桥梁架构设计，以及防御机制，如分层验证和断路器。这项工作为评估桥梁安全性提供了数据驱动的基础，并为标准化弹性跨链基础设施奠定了基础。



## **40. Sparsification Under Siege: Defending Against Poisoning Attacks in Communication-Efficient Federated Learning**

围攻下的稀疏化：在通信高效的联邦学习中防御中毒攻击 cs.CR

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2505.01454v4) [paper-pdf](http://arxiv.org/pdf/2505.01454v4)

**Authors**: Zhiyong Jin, Runhua Xu, Chao Li, Yizhong Liu, Jianxin Li, James Joshi

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy, yet it faces significant challenges in communication efficiency and vulnerability to poisoning attacks. While sparsification techniques mitigate communication overhead by transmitting only critical model parameters, they inadvertently amplify security risks: adversarial clients can exploit sparse updates to evade detection and degrade model performance. Existing defense mechanisms, designed for standard FL communication scenarios, are ineffective in addressing these vulnerabilities within sparsified FL. To bridge this gap, we propose FLARE, a novel federated learning framework that integrates sparse index mask inspection and model update sign similarity analysis to detect and mitigate poisoning attacks in sparsified FL. Extensive experiments across multiple datasets and adversarial scenarios demonstrate that FLARE significantly outperforms existing defense strategies, effectively securing sparsified FL against poisoning attacks while maintaining communication efficiency.

摘要: 联合学习（FL）支持跨分布式客户端的协作模型训练，同时保护数据隐私，但它在通信效率和中毒攻击的脆弱性方面面临着重大挑战。虽然稀疏化技术通过仅传输关键模型参数来减轻通信负担，但它们无意中放大了安全风险：对抗性客户端可以利用稀疏更新来逃避检测并降低模型性能。为标准FL通信场景设计的现有防御机制无法解决稀疏FL中的这些漏洞。为了弥合这一差距，我们提出了LGA，一个新颖的联邦学习框架，集成了稀疏索引屏蔽检查和模型更新符号相似性分析，以检测和减轻稀疏FL中的中毒攻击。跨多个数据集和对抗场景的广泛实验表明，LGA显着优于现有的防御策略，有效地保护稀疏FL免受中毒攻击，同时保持通信效率。



## **41. Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems**

对企业LLM系统的多阶段即时推理攻击 cs.CR

26 pages

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15613v1) [paper-pdf](http://arxiv.org/pdf/2507.15613v1)

**Authors**: Andrii Balashov, Olena Ponomarova, Xiaohua Zhai

**Abstract**: Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place.   We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.

摘要: 部署在企业环境中的大型语言模型（LLM）（例如，作为Microsoft 365 Copilot）面临着新颖的安全挑战。一个关键威胁是提示推理攻击：对手将看似良性的提示链接在一起，以逐渐提取机密数据。本文对企业LLM上下文中的多阶段提示推理攻击进行了全面研究。我们模拟了现实的攻击场景，其中攻击者使用温和的查询和间接提示注入来利用集成了私人公司数据的LLM。我们为这些多回合推理攻击开发了一个正式的威胁模型，并使用概率论、优化框架和信息论泄漏界限对其进行分析。这些攻击被证明可以可靠地从LLM的上下文中泄露敏感信息（例如，内部SharePoint文档或电子邮件），即使已采取标准安全措施。   我们提出并评估防御措施来对抗此类攻击，包括统计异常检测、细粒度访问控制、即时清理技术以及对LLM部署的架构修改。每个防御都有数学分析或实验模拟的支持。例如，我们在基于隐私的差异训练下推导出信息泄露的界限，并演示了一种异常检测方法，该方法可以标记具有高AUR的多回合攻击。我们还引入了一种名为“聚光灯”的方法，该方法使用输入转换来隔离不受信任的提示内容，从而将攻击成功率降低一个数量级。最后，我们为联合深度防御策略提供了正式的概念证明和经验验证。我们的工作强调，在企业环境中保护LLM需要超越单轮即时过滤，转向针对攻击和防御的整体、多阶段视角。



## **42. Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI**

统一XAI的无导扩散总管约束梯度 cs.CV

CVPR 2025 (poster), 19 pages, 5 figures

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2411.15265v2) [paper-pdf](http://arxiv.org/pdf/2411.15265v2)

**Authors**: Won Jun Kim, Hyungjin Chung, Jaemin Kim, Sangmin Lee, Byeongsu Sim, Jong Chul Ye

**Abstract**: Gradient-based methods are a prototypical family of explainability techniques, especially for image-based models. Nonetheless, they have several shortcomings in that they (1) require white-box access to models, (2) are vulnerable to adversarial attacks, and (3) produce attributions that lie off the image manifold, leading to explanations that are not actually faithful to the model and do not align well with human perception. To overcome these challenges, we introduce Derivative-Free Diffusion Manifold-Constrainted Gradients (FreeMCG), a novel method that serves as an improved basis for explainability of a given neural network than the traditional gradient. Specifically, by leveraging ensemble Kalman filters and diffusion models, we derive a derivative-free approximation of the model's gradient projected onto the data manifold, requiring access only to the model's outputs. We demonstrate the effectiveness of FreeMCG by applying it to both counterfactual generation and feature attribution, which have traditionally been treated as distinct tasks. Through comprehensive evaluation on both tasks, counterfactual explanation and feature attribution, we show that our method yields state-of-the-art results while preserving the essential properties expected of XAI tools.

摘要: 基于对象的方法是一个典型的可解释性技术家族，尤其是对于基于图像的模型。尽管如此，它们也有几个缺点，因为它们（1）需要白盒访问模型，（2）容易受到对抗性攻击，（3）产生脱离图像多管的属性，导致解释实际上不忠实于模型，并且不符合人类感知。为了克服这些挑战，我们引入了无求导扩散Manifold约束子（FreeMCG），这是一种新颖的方法，与传统梯度相比，它可以作为给定神经网络可解释性的改进基础。具体来说，通过利用集合卡尔曼过滤器和扩散模型，我们推导出投影到数据集上的模型梯度的无导逼近，仅需要访问模型的输出。我们通过将FreeMCG应用于反事实生成和特征归因来证明它的有效性，这两个传统上被视为不同的任务。通过对这两项任务、反事实解释和特征归因的全面评估，我们表明我们的方法可以产生最先进的结果，同时保留了XAI工具所期望的基本属性。



## **43. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

坏与好的传输攻击：解释和增强多模式大型语言模型之间的对抗性传输 cs.CV

This paper is accepted by ACM MM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2405.20090v5) [paper-pdf](http://arxiv.org/pdf/2405.20090v5)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.

摘要: 多模式大型语言模型（MLLM）在跨模式交互中表现出出色的性能，但它们也存在对抗性漏洞。特别是，对抗性例子的可移植性仍然是一个持续的挑战。本文具体分析了MLLM之间对抗性转移性的表现，并确定了影响这一特征的关键因素。我们发现，MLLM的可移植性存在于具有相同视觉编码器的跨LLM场景中，并指出可能影响可移植性的\underline{\textit{两个关键因素}}。我们提供了两种语义级数据增强方法：添加图像补丁（AIP）和印刷增强可移植性方法（TATM），它们增强了对抗性示例跨MLLM的可移植性。为了探索对现实世界的潜在影响，我们利用了两项可能产生负面和积极社会影响的任务：\ding{182}有害内容插入和\ding{183}信息保护。



## **44. Scaling Decentralized Learning with FLock**

使用Flock扩展分散式学习 cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15349v1) [paper-pdf](http://arxiv.org/pdf/2507.15349v1)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.

摘要: 由于集中控制的不足以及分散式方案的大量计算和通信负担，大型语言模型（LLM）的微调受到阻碍。虽然典型的标准联邦学习（FL）支持数据隐私，但中央服务器要求会创建单点攻击和中毒攻击的脆弱性。将这一方向的结果推广到异类、无信任环境中的70 B参数模型已被证明是一个巨大但未突破的瓶颈。本文介绍了Flock，这是一个用于安全高效协作LLM微调的去中心化框架。Flock将基于区块链的信任层与经济激励相结合，用安全、可审计的协议取代了中央聚合器，用于不受信任方之间的合作。我们首次对在安全、多域、去中心化的环境中微调70 B LLM进行了实证验证。我们的实验表明，Flock框架可以抵御后门中毒攻击，这些攻击会损害标准FL优化器并促进协同知识转移。由此产生的模型显示对抗性攻击成功率降低了>68%。全局模型还展示了卓越的跨域泛化能力，优于在自己的专业数据上孤立训练的模型。



## **45. Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models**

猫混淆推理LLM：推理模型的查询不可知对抗触发器 cs.CL

Accepted to CoLM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2503.01781v2) [paper-pdf](http://arxiv.org/pdf/2503.01781v2)

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.

摘要: 我们通过引入查询不可知的对抗触发器（简短、不相关的文本，当附加到数学问题时，会系统性地误导模型输出错误答案，而不改变问题的语义，来研究为逐步解决问题而训练的推理模型的稳健性。我们提出CatAttack，这是一种自动迭代攻击管道，用于在较弱、较便宜的代理模型（DeepSeek V3）上生成触发器，并成功将它们转移到DeepSeek R1和DeepSeek R1-蒸馏-Qwen-32 B等更高级的推理目标模型，导致目标模型生成错误答案的可能性增加300%以上。例如，在任何数学问题上添加“有趣的事实：猫一生中大部分时间都在睡觉”都会导致模型出错的机会增加一倍多。我们的研究结果凸显了推理模型中的关键漏洞，揭示了即使是最先进的模型仍然容易受到微妙的对抗输入的影响，从而引发了安全性和可靠性的担忧。CatAttack触发具有模型响应的数据集可在https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers上获取。



## **46. Defective Convolutional Networks**

有缺陷的卷积网络 cs.CV

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/1911.08432v3) [paper-pdf](http://arxiv.org/pdf/1911.08432v3)

**Authors**: Tiange Luo, Tianle Cai, Mengxiao Zhang, Siyu Chen, Di He, Liwei Wang

**Abstract**: Robustness of convolutional neural networks (CNNs) has gained in importance on account of adversarial examples, i.e., inputs added as well-designed perturbations that are imperceptible to humans but can cause the model to predict incorrectly. Recent research suggests that the noises in adversarial examples break the textural structure, which eventually leads to wrong predictions. To mitigate the threat of such adversarial attacks, we propose defective convolutional networks that make predictions relying less on textural information but more on shape information by properly integrating defective convolutional layers into standard CNNs. The defective convolutional layers contain defective neurons whose activations are set to be a constant function. As defective neurons contain no information and are far different from standard neurons in its spatial neighborhood, the textural features cannot be accurately extracted, and so the model has to seek other features for classification, such as the shape. We show extensive evidence to justify our proposal and demonstrate that defective CNNs can defense against black-box attacks better than standard CNNs. In particular, they achieve state-of-the-art performance against transfer-based attacks without any adversarial training being applied.

摘要: 由于对抗性示例，卷积神经网络（CNN）的鲁棒性变得越来越重要，即输入作为精心设计的扰动添加，人类无法察觉，但可能导致模型预测错误。最近的研究表明，对抗性例子中的噪音打破了文本结构，最终导致错误的预测。为了减轻此类对抗攻击的威胁，我们提出了有缺陷的卷积网络，通过将有缺陷的卷积层正确集成到标准CNN中，这些网络减少了对纹理信息的依赖，而更多地依赖形状信息来进行预测。有缺陷的卷积层包含有缺陷的神经元，其激活被设置为恒定函数。由于有缺陷的神经元不包含信息，并且在其空间附近与标准神经元相去甚远，因此无法准确提取纹理特征，因此模型必须寻找其他特征进行分类，例如形状。我们展示了大量的证据来证明我们的提议是合理的，并证明有缺陷的CNN比标准CNN更能防御黑匣子攻击。特别是，它们在不应用任何对抗训练的情况下就能获得针对基于传输的攻击的最先进性能。



## **47. ROBAD: Robust Adversary-aware Local-Global Attended Bad Actor Detection Sequential Model**

ROBAD：稳健的对手感知本地-全球参与不良行为者检测序列模型 cs.LG

15 pages, 12 tables

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.15067v1) [paper-pdf](http://arxiv.org/pdf/2507.15067v1)

**Authors**: Bing He, Mustaque Ahamad, Srijan Kumar

**Abstract**: Detecting bad actors is critical to ensure the safety and integrity of internet platforms. Several deep learning-based models have been developed to identify such users. These models should not only accurately detect bad actors, but also be robust against adversarial attacks that aim to evade detection. However, past deep learning-based detection models do not meet the robustness requirement because they are sensitive to even minor changes in the input sequence. To address this issue, we focus on (1) improving the model understanding capability and (2) enhancing the model knowledge such that the model can recognize potential input modifications when making predictions. To achieve these goals, we create a novel transformer-based classification model, called ROBAD (RObust adversary-aware local-global attended Bad Actor Detection model), which uses the sequence of user posts to generate user embedding to detect bad actors. Particularly, ROBAD first leverages the transformer encoder block to encode each post bidirectionally, thus building a post embedding to capture the local information at the post level. Next, it adopts the transformer decoder block to model the sequential pattern in the post embeddings by using the attention mechanism, which generates the sequence embedding to obtain the global information at the sequence level. Finally, to enrich the knowledge of the model, embeddings of modified sequences by mimicked attackers are fed into a contrastive-learning-enhanced classification layer for sequence prediction. In essence, by capturing the local and global information (i.e., the post and sequence information) and leveraging the mimicked behaviors of bad actors in training, ROBAD can be robust to adversarial attacks. Extensive experiments on Yelp and Wikipedia datasets show that ROBAD can effectively detect bad actors when under state-of-the-art adversarial attacks.

摘要: 检测不良行为者对于确保互联网平台的安全性和完整性至关重要。已经开发了几个基于深度学习的模型来识别此类用户。这些模型不仅应该准确地检测不良行为者，而且还应该对旨在逃避检测的对抗性攻击具有鲁棒性。然而，过去的基于深度学习的检测模型不满足稳健性要求，因为它们对输入序列中的哪怕是微小的变化也很敏感。为了解决这个问题，我们重点关注（1）提高模型理解能力和（2）增强模型知识，以便模型在做出预测时能够识别潜在的输入修改。为了实现这些目标，我们创建了一个新颖的基于转换器的分类模型，称为ROBAD（ROBust对手感知本地-全球参与的坏演员检测模型），它使用用户帖子序列来生成用户嵌入来检测坏演员。特别是，ROBAD首先利用Transformer编码器块对每个帖子进行双向编码，从而构建帖子嵌入以捕获帖子级别的本地信息。接下来，采用Transformer解码器块，利用注意机制对后嵌入中的序列模式进行建模，生成序列嵌入，以获取序列级的全局信息。最后，为了丰富模型的知识，模拟攻击者对修改后的序列的嵌入被输入对比学习增强分类层以进行序列预测。本质上，通过捕获本地和全球信息（即，发布和序列信息）并利用训练中不良行为者的模仿行为，ROBAD可以对对抗性攻击具有鲁棒性。对Yelp和维基百科数据集的大量实验表明，ROBAD可以在最先进的对抗攻击下有效地检测不良行为者。



## **48. DeRAG: Black-box Adversarial Attacks on Multiple Retrieval-Augmented Generation Applications via Prompt Injection**

DeRAG：通过提示注入对多个检索增强生成应用程序的黑匣子对抗攻击 cs.AI

Accepted by KDD Workshop on Prompt Optimization 2025

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.15042v1) [paper-pdf](http://arxiv.org/pdf/2507.15042v1)

**Authors**: Jerry Wang, Fang Yu

**Abstract**: Adversarial prompt attacks can significantly alter the reliability of Retrieval-Augmented Generation (RAG) systems by re-ranking them to produce incorrect outputs. In this paper, we present a novel method that applies Differential Evolution (DE) to optimize adversarial prompt suffixes for RAG-based question answering. Our approach is gradient-free, treating the RAG pipeline as a black box and evolving a population of candidate suffixes to maximize the retrieval rank of a targeted incorrect document to be closer to real world scenarios. We conducted experiments on the BEIR QA datasets to evaluate attack success at certain retrieval rank thresholds under multiple retrieving applications. Our results demonstrate that DE-based prompt optimization attains competitive (and in some cases higher) success rates compared to GGPP to dense retrievers and PRADA to sparse retrievers, while using only a small number of tokens (<=5 tokens) in the adversarial suffix. Furthermore, we introduce a readability-aware suffix construction strategy, validated by a statistically significant reduction in MLM negative log-likelihood with Welch's t-test. Through evaluations with a BERT-based adversarial suffix detector, we show that DE-generated suffixes evade detection, yielding near-chance detection accuracy.

摘要: 对抗提示攻击可以通过重新排序检索增强生成（RAG）系统以产生错误的输出来显着改变它们的可靠性。在本文中，我们提出了一种应用差异进化（DE）来优化基于RAG的问答的对抗性提示后缀的新方法。我们的方法是无梯度的，将RAG管道视为黑匣子，并进化出一系列候选后缀，以最大化目标错误文档的检索排名，使其更接近现实世界场景。我们对BEIR QA数据集进行了实验，以评估多个检索应用程序下特定检索等级阈值下的攻击成功率。我们的结果表明，与GDPP对密集检索器和PRADA对稀疏检索器相比，基于DE的即时优化获得了有竞争力的（在某些情况下更高）成功率，同时在对抗性后缀中仅使用少量令牌（<=5个令牌）。此外，我们引入了一种可读写性感知的后缀构建策略，该策略通过韦尔奇t检验MLM负对log似然性的统计学显着降低来验证。通过使用基于BERT的对抗性后缀检测器进行评估，我们表明DE生成的后缀可以逃避检测，从而产生近乎偶然的检测准确性。



## **49. Adversarial Destabilization Attacks to Direct Data-Driven Control**

直接数据驱动控制的对抗性去稳定攻击 eess.SY

15 pages

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14863v1) [paper-pdf](http://arxiv.org/pdf/2507.14863v1)

**Authors**: Hampei Sasahara

**Abstract**: This study investigates the vulnerability of direct data-driven control methods, specifically for the linear quadratic regulator problem, to adversarial perturbations in collected data used for controller synthesis. We consider stealthy attacks that subtly manipulate offline-collected data to destabilize the resulting closed-loop system while evading detection. To generate such perturbations, we propose the Directed Gradient Sign Method (DGSM) and its iterative variant (I-DGSM), adaptations of the fast gradient sign method originally developed for neural networks, which align perturbations with the gradient of the spectral radius of the closed-loop matrix to reduce stability. A key contribution is an efficient gradient computation technique based on implicit differentiation through the Karush-Kuhn-Tucker conditions of the underlying semidefinite program, enabling scalable and exact gradient evaluation without repeated optimization computations. To defend against these attacks, we propose two defense strategies: a regularization-based approach that enhances robustness by suppressing controller sensitivity to data perturbations and a robust data-driven control approach that guarantees closed-loop stability within bounded perturbation sets. Extensive numerical experiments on benchmark systems show that adversarial perturbations with magnitudes up to ten times smaller than random noise can destabilize controllers trained on corrupted data and that the proposed defense strategies effectively mitigate attack success rates while maintaining control performance. Additionally, we evaluate attack transferability under partial knowledge scenarios, highlighting the practical importance of protecting training data confidentiality.

摘要: 本研究调查了直接数据驱动控制方法（特别是线性二次调节器问题）对用于控制器综合的收集数据中对抗性扰动的脆弱性。我们考虑的是隐蔽攻击，这些攻击巧妙地操纵离线收集的数据，以破坏由此产生的闭环系统的稳定性，同时逃避检测。为了产生此类扰动，我们提出了有向梯度符号法（DGSM）及其迭代变体（I-DGSM），这是最初为神经网络开发的快速梯度符号法的改编，它将扰动与闭环矩阵的谱半径的梯度对齐以降低稳定性。一个关键贡献是一种基于通过底层半定程序的Karush-Kuhn-Tucker条件进行隐式求导的高效梯度计算技术，无需重复优化计算即可实现可扩展且准确的梯度评估。为了抵御这些攻击，我们提出了两种防御策略：一种基于正规化的方法，通过抑制控制器对数据扰动的敏感性来增强鲁棒性，另一种鲁棒的数据驱动控制方法，保证有界扰动集中的闭环稳定性。对基准系统进行的大量数值实验表明，幅度比随机噪音小十倍的对抗性扰动可能会破坏根据损坏数据训练的控制器的稳定性，并且提出的防御策略有效地降低了攻击成功率，同时保持控制性能。此外，我们还评估了部分知识场景下的攻击可转移性，强调了保护训练数据机密性的实际重要性。



## **50. Data-Plane Telemetry to Mitigate Long-Distance BGP Hijacks**

数据平面遥感缓解长距离BNP劫持 cs.NI

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14842v1) [paper-pdf](http://arxiv.org/pdf/2507.14842v1)

**Authors**: Satadal Sengupta, Hyojoon Kim, Daniel Jubas, Maria Apostolaki, Jennifer Rexford

**Abstract**: Poor security of Internet routing enables adversaries to divert user data through unintended infrastructures (hijack). Of particular concern -- and the focus of this paper -- are cases where attackers reroute domestic traffic through foreign countries, exposing it to surveillance, bypassing legal privacy protections, and posing national security threats. Efforts to detect and mitigate such attacks have focused primarily on the control plane while data-plane signals remain largely overlooked. In particular, change in propagation delay caused by rerouting offers a promising signal: the change is unavoidable and the increased propagation delay is directly observable from the affected networks. In this paper, we explore the practicality of using delay variations for hijack detection, addressing two key questions: (1) What coverage can this provide, given its heavy dependence on the geolocations of the sender, receiver, and adversary? and (2) Can an always-on latency-based detection system be deployed without disrupting normal network operations? We observe that for 86% of victim-attacker country pairs in the world, mid-attack delays exceed pre-attack delays by at least 25% in real deployments, making delay-based hijack detection promising. To demonstrate practicality, we design HiDe, which reliably detects delay surges from long-distance hijacks at line rate. We measure HiDe's accuracy and false-positive rate on real-world data and validate it with ethically conducted hijacks.

摘要: 互联网路由的安全性较差使对手能够通过意外的基础设施（劫持）转移用户数据。特别令人担忧的情况--也是本文的重点--是攻击者通过外国重新路由国内流量，使其受到监视，绕过法律隐私保护，并构成国家安全威胁。检测和减轻此类攻击的努力主要集中在控制平面上，而数据平面信号在很大程度上仍然被忽视。特别是，重新路由引起的传播延迟变化提供了一个有希望的信号：这种变化是不可避免的，并且传播延迟的增加可以从受影响的网络中直接观察到。在本文中，我们探讨了使用延迟变化的劫持检测的实用性，解决两个关键问题：（1）这可以提供什么样的覆盖范围，鉴于其严重依赖于发送者，接收器和对手的地理位置？以及（2）可以在不中断正常网络操作的情况下部署始终在线的基于延迟的检测系统吗？我们观察到，对于世界上86%的受害者-攻击者国家对，在实际部署中，攻击中期延迟超过攻击前延迟至少25%，这使得基于延迟的劫持检测很有希望。为了证明实用性，我们设计了HiDe，它可以以线速可靠地检测来自长距离劫持的延迟浪涌。我们在现实世界数据上衡量HiDe的准确性和假阳性率，并通过道德劫持来验证它。



