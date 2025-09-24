# Latest Adversarial Attack Papers
**update at 2025-09-24 09:11:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AI-Generated Text is Non-Stationary: Detection via Temporal Tomography**

人工智能生成的文本是非静止的：通过时间断层扫描检测 cs.CL

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2508.01754v2) [paper-pdf](http://arxiv.org/pdf/2508.01754v2)

**Authors**: Alva West, Yixuan Weng, Minjun Zhu, Luodan Zhang, Zhen Lin, Guangsheng Bao, Yue Zhang

**Abstract**: The field of AI-generated text detection has evolved from supervised classification to zero-shot statistical analysis. However, current approaches share a fundamental limitation: they aggregate token-level measurements into scalar scores, discarding positional information about where anomalies occur. Our empirical analysis reveals that AI-generated text exhibits significant non-stationarity, statistical properties vary by 73.8\% more between text segments compared to human writing. This discovery explains why existing detectors fail against localized adversarial perturbations that exploit this overlooked characteristic. We introduce Temporal Discrepancy Tomography (TDT), a novel detection paradigm that preserves positional information by reformulating detection as a signal processing task. TDT treats token-level discrepancies as a time-series signal and applies Continuous Wavelet Transform to generate a two-dimensional time-scale representation, capturing both the location and linguistic scale of statistical anomalies. On the RAID benchmark, TDT achieves 0.855 AUROC (7.1\% improvement over the best baseline). More importantly, TDT demonstrates robust performance on adversarial tasks, with 14.1\% AUROC improvement on HART Level 2 paraphrasing attacks. Despite its sophisticated analysis, TDT maintains practical efficiency with only 13\% computational overhead. Our work establishes non-stationarity as a fundamental characteristic of AI-generated text and demonstrates that preserving temporal dynamics is essential for robust detection.

摘要: 人工智能生成的文本检测领域已经从监督分类发展到零镜头统计分析。然而，当前的方法都有一个根本性的局限性：它们将标记级测量结果聚合为纯量分数，丢弃有关异常发生位置的位置信息。我们的实证分析表明，人工智能生成的文本表现出显着的非平稳性，与人类写作相比，文本片段之间的统计属性差异大73.8%。这一发现解释了为什么现有的检测器无法对抗利用这一被忽视的特征的局部对抗性扰动。我们引入了时间离散断层扫描（TDT），这是一种新型检测范式，通过将检测重新定义为信号处理任务来保留位置信息。TDT将标记级别差异视为时间序列信号，并应用连续子波变换来生成二维时间尺度表示，从而捕获统计异常的位置和语言尺度。在磁盘基准测试中，TDT达到了0.855 AUROC（比最佳基线提高了7.1%）。更重要的是，TDT在对抗性任务上表现出稳健的性能，在HART2级重述攻击方面提高了14.1% AUROC。尽管TDT分析复杂，但只需13%的计算费用即可保持实际效率。我们的工作将非平稳性确立为人工智能生成文本的基本特征，并证明保留时间动态对于鲁棒检测至关重要。



## **2. Algorithms for Adversarially Robust Deep Learning**

对抗鲁棒深度学习算法 cs.LG

PhD thesis

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.19100v1) [paper-pdf](http://arxiv.org/pdf/2509.19100v1)

**Authors**: Alexander Robey

**Abstract**: Given the widespread use of deep learning models in safety-critical applications, ensuring that the decisions of such models are robust against adversarial exploitation is of fundamental importance. In this thesis, we discuss recent progress toward designing algorithms that exhibit desirable robustness properties. First, we discuss the problem of adversarial examples in computer vision, for which we introduce new technical results, training paradigms, and certification algorithms. Next, we consider the problem of domain generalization, wherein the task is to train neural networks to generalize from a family of training distributions to unseen test distributions. We present new algorithms that achieve state-of-the-art generalization in medical imaging, molecular identification, and image classification. Finally, we study the setting of jailbreaking large language models (LLMs), wherein an adversarial user attempts to design prompts that elicit objectionable content from an LLM. We propose new attacks and defenses, which represent the frontier of progress toward designing robust language-based agents.

摘要: 鉴于深度学习模型在安全关键应用中的广泛使用，确保此类模型的决策针对对抗性剥削具有鲁棒性至关重要。在这篇论文中，我们讨论了设计具有理想鲁棒性的算法的最新进展。首先，我们讨论计算机视觉中的对抗性示例问题，为此我们介绍了新的技术成果、训练范式和认证算法。接下来，我们考虑领域概括问题，其中的任务是训练神经网络将一系列训练分布推广到不可见的测试分布。我们提出了新算法，可以在医学成像、分子识别和图像分类方面实现最先进的概括。最后，我们研究了越狱大型语言模型（LLM）的设置，其中敌对用户试图设计从LLM中引出令人反感的内容的提示。我们提出了新的攻击和防御，这代表了设计稳健的基于语言的代理的进展前沿。



## **3. Latent Danger Zone: Distilling Unified Attention for Cross-Architecture Black-box Attacks**

潜在危险区：提炼对跨体系结构黑匣子攻击的统一注意力 cs.LG

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.19044v1) [paper-pdf](http://arxiv.org/pdf/2509.19044v1)

**Authors**: Yang Li, Chenyu Wang, Tingrui Wang, Yongwei Wang, Haonan Li, Zhunga Liu, Quan Pan

**Abstract**: Black-box adversarial attacks remain challenging due to limited access to model internals. Existing methods often depend on specific network architectures or require numerous queries, resulting in limited cross-architecture transferability and high query costs. To address these limitations, we propose JAD, a latent diffusion model framework for black-box adversarial attacks. JAD generates adversarial examples by leveraging a latent diffusion model guided by attention maps distilled from both a convolutional neural network (CNN) and a Vision Transformer (ViT) models. By focusing on image regions that are commonly sensitive across architectures, this approach crafts adversarial perturbations that transfer effectively between different model types. This joint attention distillation strategy enables JAD to be architecture-agnostic, achieving superior attack generalization across diverse models. Moreover, the generative nature of the diffusion framework yields high adversarial sample generation efficiency by reducing reliance on iterative queries. Experiments demonstrate that JAD offers improved attack generalization, generation efficiency, and cross-architecture transferability compared to existing methods, providing a promising and effective paradigm for black-box adversarial attacks.

摘要: 由于对模型内部内容的访问有限，黑匣子对抗攻击仍然具有挑战性。现有的方法通常取决于特定的网络架构或需要大量查询，导致跨架构可移植性有限且查询成本高。为了解决这些限制，我们提出了JAD，这是一种用于黑匣子对抗攻击的潜在扩散模型框架。JAD通过利用由从卷积神经网络（CNN）和Vision Transformer（ViT）模型中提取的注意力图引导的潜在扩散模型来生成对抗性示例。通过专注于跨架构通常敏感的图像区域，这种方法可以创建在不同模型类型之间有效转移的对抗性扰动。这种联合注意力蒸馏策略使JAD能够与架构无关，在不同模型之间实现卓越的攻击概括性。此外，扩散框架的生成性质通过减少对迭代查询的依赖来产生高对抗性样本生成效率。实验表明，与现有方法相比，JAD提供了改进的攻击概括性、生成效率和跨架构可移植性，为黑匣子对抗攻击提供了一个有希望且有效的范式。



## **4. Towards Privacy-Aware Bayesian Networks: A Credal Approach**

迈向具有隐私意识的Bayesian网络：一种信条方法 cs.LG

Accepted at ECAI2025 conference, 20 pages, 1 figure

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18949v1) [paper-pdf](http://arxiv.org/pdf/2509.18949v1)

**Authors**: Niccolò Rocchi, Fabio Stella, Cassio de Campos

**Abstract**: Bayesian networks (BN) are probabilistic graphical models that enable efficient knowledge representation and inference. These have proven effective across diverse domains, including healthcare, bioinformatics and economics. The structure and parameters of a BN can be obtained by domain experts or directly learned from available data. However, as privacy concerns escalate, it becomes increasingly critical for publicly released models to safeguard sensitive information in training data. Typically, released models do not prioritize privacy by design. In particular, tracing attacks from adversaries can combine the released BN with auxiliary data to determine whether specific individuals belong to the data from which the BN was learned. State-of-the-art protection tecniques involve introducing noise into the learned parameters. While this offers robust protection against tracing attacks, it significantly impacts the model's utility, in terms of both the significance and accuracy of the resulting inferences. Hence, high privacy may be attained at the cost of releasing a possibly ineffective model. This paper introduces credal networks (CN) as a novel solution for balancing the model's privacy and utility. After adapting the notion of tracing attacks, we demonstrate that a CN enables the masking of the learned BN, thereby reducing the probability of successful attacks. As CNs are obfuscated but not noisy versions of BNs, they can achieve meaningful inferences while safeguarding privacy. Moreover, we identify key learning information that must be concealed to prevent attackers from recovering the underlying BN. Finally, we conduct a set of numerical experiments to analyze how privacy gains can be modulated by tuning the CN hyperparameters. Our results confirm that CNs provide a principled, practical, and effective approach towards the development of privacy-aware probabilistic graphical models.

摘要: Bayesian网络（BN）是概率图形模型，可以实现高效的知识表示和推理。事实证明，这些措施在医疗保健、生物信息学和经济学等各个领域都有效。BN的结构和参数可以由领域专家获取或直接从可用数据中学习。然而，随着隐私问题的升级，公开发布的模型保护训练数据中的敏感信息变得越来越重要。通常，发布的模型不会根据设计优先考虑隐私。特别是，追踪来自对手的攻击可以将发布的BN与辅助数据结合起来，以确定特定个人是否属于从中学习BN的数据。最先进的保护技术涉及将噪音引入学习参数。虽然这提供了针对跟踪攻击的强大保护，但它会显着影响模型的实用性，无论是结果推论的重要性还是准确性。因此，高隐私性可能会以发布可能无效的模型为代价。本文引入了credal网络（CN）作为平衡模型隐私性和实用性的新型解决方案。在适应跟踪攻击的概念后，我们证明CN可以屏蔽学习到的BN，从而降低成功攻击的可能性。由于CN是模糊的，但不是BN的嘈杂版本，因此它们可以在保护隐私的同时实现有意义的推断。此外，我们确定了必须隐藏的关键学习信息，以防止攻击者恢复底层BN。最后，我们进行了一组数值实验来分析如何通过调整CN超参数来调制隐私增益。我们的研究结果证实，CN提供了一个原则性的，实用的，有效的方法对发展的隐私意识的概率图形模型。



## **5. Generic Adversarial Smart Contract Detection with Semantics and Uncertainty-Aware LLM**

具有语义和不确定性意识的通用对抗智能合同检测LLM cs.CR

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18934v1) [paper-pdf](http://arxiv.org/pdf/2509.18934v1)

**Authors**: Yating Liu, Xing Su, Hao Wu, Sijin Li, Yuxi Cheng, Fengyuan Xu, Sheng Zhong

**Abstract**: Adversarial smart contracts, mostly on EVM-compatible chains like Ethereum and BSC, are deployed as EVM bytecode to exploit vulnerable smart contracts typically for financial gains. Detecting such malicious contracts at the time of deployment is an important proactive strategy preventing loss from victim contracts. It offers a better cost-benefit than detecting vulnerabilities on diverse potential victims. However, existing works are not generic with limited detection types and effectiveness due to imbalanced samples, while the emerging LLM technologies, which show its potentials in generalization, have two key problems impeding its application in this task: hard digestion of compiled-code inputs, especially those with task-specific logic, and hard assessment of LLMs' certainty in their binary answers, i.e., yes-or-no answers. Therefore, we propose a generic adversarial smart contracts detection framework FinDet, which leverages LLMs with two enhancements addressing above two problems. FinDet takes as input only the EVM-bytecode contracts and identifies adversarial ones among them with high balanced accuracy. The first enhancement extracts concise semantic intentions and high-level behavioral logic from the low-level bytecode inputs, unleashing the LLM reasoning capability restricted by the task input. The second enhancement probes and measures the LLM uncertainty to its multi-round answering to the same query, improving the LLM answering robustness for binary classifications required by the task output. Our comprehensive evaluation shows that FinDet achieves a BAC of 0.9223 and a TPR of 0.8950, significantly outperforming existing baselines. It remains robust under challenging conditions including unseen attack patterns, low-data settings, and feature obfuscation. FinDet detects all 5 public and 20+ unreported adversarial contracts in a 10-day real-world test, confirmed manually.

摘要: 对抗性智能合同主要在以太坊和BSC等与ESM兼容的链上，被部署为EVM字节码，以利用脆弱的智能合同，通常是为了经济利益。在部署时检测此类恶意合同是防止受害者合同损失的重要主动策略。它比检测不同潜在受害者的漏洞提供更好的成本效益。然而，现有的作品并不通用，由于样本不平衡，检测类型和有效性受到限制，而新兴的LLM技术在概括方面表现出了潜力，但存在两个关键问题阻碍其在这项任务中的应用：难以消化已编译代码输入，尤其是那些具有特定任务逻辑的代码输入，以及难以评估LLM二进制答案的确定性，即是或否的答案。因此，我们提出了一个通用的对抗智能合同检测框架FinDet，该框架利用LLM，并通过两项增强来解决上述两个问题。FinDet仅将EVM-字节码合同作为输入，并以高度平衡的准确性识别其中的对抗合同。第一个增强从低级字节码输入中提取简洁的语义意图和高级行为逻辑，释放受任务输入限制的LLM推理能力。第二个增强探测和测量LLM对同一查询的多轮回答的不确定性，提高了任务输出所需的二进制分类的LLM回答稳健性。我们的综合评估显示，FinDet的BAT为0.9223，TLR为0.8950，显着优于现有基线。它在具有挑战性的条件下保持稳健，包括不可见的攻击模式、低数据设置和功能混淆。FinDet在为期10天的现实世界测试中检测到所有5个公开和20多个未报告的对抗性合同，并手动确认。



## **6. Enhancing the Effectiveness and Durability of Backdoor Attacks in Federated Learning through Maximizing Task Distinction**

通过最大化任务区分度提高联邦学习中后门攻击的有效性和持久性 cs.LG

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18904v1) [paper-pdf](http://arxiv.org/pdf/2509.18904v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Federated learning allows multiple participants to collaboratively train a central model without sharing their private data. However, this distributed nature also exposes new attack surfaces. In particular, backdoor attacks allow attackers to implant malicious behaviors into the global model while maintaining high accuracy on benign inputs. Existing attacks usually rely on fixed patterns or adversarial perturbations as triggers, which tightly couple the main and backdoor tasks. This coupling makes them vulnerable to dilution by honest updates and limits their persistence under federated defenses. In this work, we propose an approach to decouple the backdoor task from the main task by dynamically optimizing the backdoor trigger within a min-max framework. The inner layer maximizes the performance gap between poisoned and benign samples, ensuring that the contributions of benign users have minimal impact on the backdoor. The outer process injects the adaptive triggers into the local model. We evaluate our method on both computer vision and natural language tasks, and compare it with six backdoor attack methods under six defense algorithms. Experimental results show that our method achieves good attack performance and can be easily integrated into existing backdoor attack techniques.

摘要: 联合学习允许多个参与者协作训练中央模型，而无需共享他们的私人数据。然而，这种分布式性质也暴露了新的攻击面。特别是，后门攻击允许攻击者将恶意行为植入到全局模型中，同时保持良性输入的高准确性。现有的攻击通常依赖于固定模式或对抗性扰动作为触发器，这些触发器将主要任务和后门任务紧密结合在一起。这种耦合使它们容易受到诚实更新的稀释，并限制了它们在联邦防御下的持久性。在这项工作中，我们提出了一种通过在最小-最大框架内动态优化后门触发器来将后门任务与主要任务脱钩的方法。内层最大化了有毒样本和良性样本之间的性能差距，确保良性用户的贡献对后门的影响最小。外部流程将自适应触发器注入到本地模型中。我们在计算机视觉和自然语言任务上评估了我们的方法，并将其与六种防御算法下的六种后门攻击方法进行比较。实验结果表明，我们的方法取得了良好的攻击性能，并且可以轻松集成到现有的后门攻击技术中。



## **7. Attack for Defense: Adversarial Agents for Point Prompt Optimization Empowering Segment Anything Model**

防御攻击：点提示优化的对抗性代理为分段任何模型提供支持 cs.CV

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18891v1) [paper-pdf](http://arxiv.org/pdf/2509.18891v1)

**Authors**: Xueyu Liu, Xiaoyi Zhang, Guangze Shi, Meilin Liu, Yexin Lai, Yongfei Wu, Mingqiang Wei

**Abstract**: Prompt quality plays a critical role in the performance of the Segment Anything Model (SAM), yet existing approaches often rely on heuristic or manually crafted prompts, limiting scalability and generalization. In this paper, we propose Point Prompt Defender, an adversarial reinforcement learning framework that adopts an attack-for-defense paradigm to automatically optimize point prompts. We construct a task-agnostic point prompt environment by representing image patches as nodes in a dual-space graph, where edges encode both physical and semantic distances. Within this environment, an attacker agent learns to activate a subset of prompts that maximally degrade SAM's segmentation performance, while a defender agent learns to suppress these disruptive prompts and restore accuracy. Both agents are trained using Deep Q-Networks with a reward signal based on segmentation quality variation. During inference, only the defender is deployed to refine arbitrary coarse prompt sets, enabling enhanced SAM segmentation performance across diverse tasks without retraining. Extensive experiments show that Point Prompt Defender effectively improves SAM's robustness and generalization, establishing a flexible, interpretable, and plug-and-play framework for prompt-based segmentation.

摘要: 提示质量在Segment Anything模型（Sam）的性能中发挥着关键作用，但现有方法通常依赖于启发式或手动制作的提示，从而限制了可扩展性和概括性。在本文中，我们提出了Point Proment Defender，这是一个对抗性强化学习框架，采用攻击换防御范式来自动优化Point Promise。我们通过将图像补丁表示为双空间图中的节点，构建一个任务不可知的点提示环境，其中边对物理和语义距离进行编码。在此环境中，攻击者代理学会激活提示子集，以最大限度地降低Sam的分段性能，而防御者代理则学会抑制这些破坏性提示并恢复准确性。两个代理都使用深度Q网络进行训练，并具有基于分段质量变化的奖励信号。在推理过程中，仅部署防御者来细化任意粗略提示集，从而在不同任务中增强Sam分段性能，无需重新训练。大量实验表明，Point Proment Defender有效提高了Sam的鲁棒性和通用性，为基于预算的分割建立了灵活、可解释、即插即用的框架。



## **8. Diversity Boosts AI-Generated Text Detection**

多样性增强人工智能生成文本检测 cs.CL

Project Webpage: https://diveye.vercel.app/

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18880v1) [paper-pdf](http://arxiv.org/pdf/2509.18880v1)

**Authors**: Advik Raj Basani, Pin-Yu Chen

**Abstract**: Detecting AI-generated text is an increasing necessity to combat misuse of LLMs in education, business compliance, journalism, and social media, where synthetic fluency can mask misinformation or deception. While prior detectors often rely on token-level likelihoods or opaque black-box classifiers, these approaches struggle against high-quality generations and offer little interpretability. In this work, we propose DivEye, a novel detection framework that captures how unpredictability fluctuates across a text using surprisal-based features. Motivated by the observation that human-authored text exhibits richer variability in lexical and structural unpredictability than LLM outputs, DivEye captures this signal through a set of interpretable statistical features. Our method outperforms existing zero-shot detectors by up to 33.2% and achieves competitive performance with fine-tuned baselines across multiple benchmarks. DivEye is robust to paraphrasing and adversarial attacks, generalizes well across domains and models, and improves the performance of existing detectors by up to 18.7% when used as an auxiliary signal. Beyond detection, DivEye provides interpretable insights into why a text is flagged, pointing to rhythmic unpredictability as a powerful and underexplored signal for LLM detection.

摘要: 检测人工智能生成的文本对于打击教育、商业合规、新闻和社交媒体中LLM的滥用越来越有必要，因为合成流利性可以掩盖错误信息或欺骗。虽然以前的检测器通常依赖于标记级可能性或不透明的黑匣子分类器，但这些方法难以应对高质量的世代，并且几乎没有提供可解释性。在这项工作中，我们提出了DivEye，这是一个新颖的检测框架，可以使用基于日历的特征来捕捉文本中不可预测性的波动程度。由于人类创作的文本在词汇和结构不可预测性方面表现出比LLM输出更丰富的变异性，DivEye通过一组可解释的统计特征捕捉这一信号。我们的方法比现有的零激发检测器高出33.2%，并通过跨多个基准进行微调的基线来实现有竞争力的性能。DivEye对重述和对抗攻击具有鲁棒性，可以很好地在各个领域和模型中进行推广，并在用作辅助信号时将现有检测器的性能提高高达18.7%。除了检测之外，DivEye还提供了有关文本为何被标记的可解释见解，指出节奏的不可预测性是LLM检测的一个强大且未充分探索的信号。



## **9. Fix your downsampling ASAP! Be natively more robust via Aliasing and Spectral Artifact free Pooling**

尽快修复您的下采样！通过别名和无频谱池化使本机更加稳健 cs.CV

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2307.09804v2) [paper-pdf](http://arxiv.org/pdf/2307.09804v2)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstract**: Convolutional Neural Networks (CNNs) are successful in various computer vision tasks. From an image and signal processing point of view, this success is counter-intuitive, as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. the Sampling Theorem in their downsampling operations. This issue has been broadly neglected until recent work in the context of adversarial attacks and distribution shifts showed that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by bandlimit-violating downsampling. As a remedy, we propose an alias-free downsampling operation in the frequency domain, denoted Frequency Low Cut Pooling (FLC Pooling) which we further extend to Aliasing and Sinc Artifact-free Pooling (ASAP). ASAP is alias-free and removes further artifacts from sinc-interpolation. Our experimental evaluation on ImageNet-1k, ImageNet-C and CIFAR datasets on various CNN architectures demonstrates that networks using FLC Pooling and ASAP as downsampling methods learn more stable features as measured by their robustness against common corruptions and adversarial attacks, while maintaining a clean accuracy similar to the respective baseline models.

摘要: 卷积神经网络（CNN）在各种计算机视觉任务中取得了成功。从图像和信号处理的角度来看，这种成功是违反直觉的，因为大多数CNN固有的空间金字塔设计显然违反了基本的信号处理定律，即下采样操作中的采样定理。这个问题一直被广泛忽视，直到最近在对抗性攻击和分布转变背景下的工作表明，CNN的脆弱性与违反带宽限制的下采样引起的混叠伪影之间存在很强的相关性。作为补救措施，我们提出了频域中的无别名下采样操作，称为频率低切池（FLC池），我们进一步将其扩展到别名和Sinc无伪影池（ASAP）。ASAP无别名，并从sinc插值中删除了进一步的伪影。我们对各种CNN架构上的ImageNet-1 k、ImageNet-C和CIFAR数据集进行的实验评估表明，使用FLC Pooling和ASAP作为下采样方法的网络可以学习更稳定的特征，这是通过其针对常见破坏和对抗性攻击的鲁棒性来衡量的，同时保持与各自的基线模型类似的清晰准确性。



## **10. TriFusion-AE: Language-Guided Depth and LiDAR Fusion for Robust Point Cloud Processing**

TriFusion-AE：激光引导深度和LiDART融合，实现稳健的点云处理 cs.CV

39th Conference on Neural Information Processing Systems (NeurIPS  2025) Workshop

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18743v1) [paper-pdf](http://arxiv.org/pdf/2509.18743v1)

**Authors**: Susmit Neogi

**Abstract**: LiDAR-based perception is central to autonomous driving and robotics, yet raw point clouds remain highly vulnerable to noise, occlusion, and adversarial corruptions. Autoencoders offer a natural framework for denoising and reconstruction, but their performance degrades under challenging real-world conditions. In this work, we propose TriFusion-AE, a multimodal cross-attention autoencoder that integrates textual priors, monocular depth maps from multi-view images, and LiDAR point clouds to improve robustness. By aligning semantic cues from text, geometric (depth) features from images, and spatial structure from LiDAR, TriFusion-AE learns representations that are resilient to stochastic noise and adversarial perturbations. Interestingly, while showing limited gains under mild perturbations, our model achieves significantly more robust reconstruction under strong adversarial attacks and heavy noise, where CNN-based autoencoders collapse. We evaluate on the nuScenes-mini dataset to reflect realistic low-data deployment scenarios. Our multimodal fusion framework is designed to be model-agnostic, enabling seamless integration with any CNN-based point cloud autoencoder for joint representation learning.

摘要: 基于激光雷达的感知是自动驾驶和机器人技术的核心，但原始点云仍然极易受到噪音、遮挡和对抗腐蚀的影响。自动编码器提供了一个自然的去噪和重建框架，但在具有挑战性的现实条件下，它们的性能会下降。在这项工作中，我们提出了TriFusion-AE，这是一种多模式交叉注意自动编码器，它集成了文本先验、来自多视图图像的单目深度图和LiDART点云，以提高鲁棒性。通过对齐来自文本的语义线索、来自图像的几何（深度）特征和来自LiDART的空间结构，TriFusion-AE学习能够抵御随机噪音和对抗性扰动的表示。有趣的是，虽然我们的模型在轻微扰动下表现出有限的收益，但在强对抗攻击和强噪音下（基于CNN的自动编码器崩溃）实现了显着更稳健的重建。我们对nuScenes-mini数据集进行评估，以反映现实的低数据部署场景。我们的多模式融合框架旨在与模型无关，能够与任何基于CNN的点云自动编码器无缝集成，以进行联合表示学习。



## **11. Automating Steering for Safe Multimodal Large Language Models**

安全多模式大型语言模型的自动转向 cs.CL

EMNLP 2025 Main Conference. 23 pages (8+ for main); 25 figures; 1  table

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2507.13255v3) [paper-pdf](http://arxiv.org/pdf/2507.13255v3)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.

摘要: 多模式大型语言模型（MLLM）的最新进展释放了强大的跨模式推理能力，但也提出了新的安全问题，特别是在面对对抗性多模式输入时。为了提高MLLM在推理过程中的安全性，我们引入了模块化和自适应的推理时干预技术AutoSteer，无需对底层模型进行任何微调。AutoSteer包含三个核心组件：（1）新型安全意识评分（SAS），自动识别模型内部层之间最安全相关的区别;（2）自适应安全探测器，经过训练以估计中间表示有毒输出的可能性;（3）轻量级拒绝头，当检测到安全风险时，它会选择性地干预以调节发电。LLaVa-OG和Chameleon在各种安全关键基准上的实验表明，AutoSteer显着降低了文本、视觉和跨模式威胁的攻击成功率（ASB），同时保持了一般能力。这些发现将AutoSteer定位为一个实用、可解释且有效的框架，用于更安全地部署多模式人工智能系统。



## **12. Examining I2P Resilience: Effect of Centrality-based Attack**

检查I2P弹性：基于中心性的攻击的影响 cs.CR

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18572v1) [paper-pdf](http://arxiv.org/pdf/2509.18572v1)

**Authors**: Kemi Akanbi, Sunkanmi Oluwadare, Jess Kropczynski, Jacques Bou Abdo

**Abstract**: This study examines the robustness of I2P, a well-regarded anonymous and decentralized peer-to-peer network designed to ensure anonymity, confidentiality, and circumvention of censorship. Unlike its more widely researched counterpart, TOR, I2P's resilience has received less scholarly attention. Employing network analysis, this research evaluates I2P's susceptibility to adversarial percolation. By utilizing the degree centrality as a measure of nodes' influence in the network, the finding suggests the network is vulnerable to targeted disruptions. Before percolation, the network exhibited a density of 0.01065443 and an average path length of 6.842194. At the end of the percolation process, the density decreased by approximately 10%, and the average path length increased by 33%, indicating a decline in efficiency and connectivity. These results highlight that even decentralized networks, such as I2P, exhibit structural fragility under targeted attacks, emphasizing the need for improved design strategies to enhance resilience against adversarial disruptions.

摘要: 这项研究考察了I2P的稳健性，I2P是一个广受好评的匿名和去中心化的点对点网络，旨在确保匿名性、保密性和规避审查制度。与研究更广泛的同行TOR不同，I2P的弹性受到的学术关注较少。这项研究利用网络分析评估了I2P对对抗渗透的敏感性。通过利用度中心性作为节点在网络中影响力的衡量标准，研究结果表明网络很容易受到有针对性的中断的影响。在渗透之前，网络的密度为0.01065443，平均路径长度为6.842194。渗透过程结束时，密度下降了约10%，平均路径长度增加了33%，表明效率和连通性下降。这些结果强调，即使是I2P等去中心化网络，在有针对性的攻击下也表现出结构脆弱性，强调需要改进设计策略，以增强针对对抗性中断的弹性。



## **13. SEGA: A Transferable Signed Ensemble Gaussian Black-Box Attack against No-Reference Image Quality Assessment Models**

SEGA：针对无参考图像质量评估模型的可传输签名集合高斯黑匣子攻击 cs.CV

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18546v1) [paper-pdf](http://arxiv.org/pdf/2509.18546v1)

**Authors**: Yujia Liu, Dingquan Li, Tiejun Huang

**Abstract**: No-Reference Image Quality Assessment (NR-IQA) models play an important role in various real-world applications. Recently, adversarial attacks against NR-IQA models have attracted increasing attention, as they provide valuable insights for revealing model vulnerabilities and guiding robust system design. Some effective attacks have been proposed against NR-IQA models in white-box settings, where the attacker has full access to the target model. However, these attacks often suffer from poor transferability to unknown target models in more realistic black-box scenarios, where the target model is inaccessible. This work makes the first attempt to address the challenge of low transferability in attacking NR-IQA models by proposing a transferable Signed Ensemble Gaussian black-box Attack (SEGA). The main idea is to approximate the gradient of the target model by applying Gaussian smoothing to source models and ensembling their smoothed gradients. To ensure the imperceptibility of adversarial perturbations, SEGA further removes inappropriate perturbations using a specially designed perturbation filter mask. Experimental results on the CLIVE dataset demonstrate the superior transferability of SEGA, validating its effectiveness in enabling successful transfer-based black-box attacks against NR-IQA models.

摘要: 无参考图像质量评估（NR-IQA）模型在各种现实应用中发挥着重要作用。最近，针对NR-IQA模型的对抗攻击引起了越来越多的关注，因为它们为揭示模型漏洞和指导稳健的系统设计提供了宝贵的见解。在白盒环境中，针对NR-IQA模型提出了一些有效的攻击，其中攻击者可以完全访问目标模型。然而，在更现实的黑匣子场景中，这些攻击通常会出现向未知目标模型的可移植性较差的情况，其中目标模型不可访问。针对NR-IQA模型的可转移性差的问题，提出了一种可转移的签名包围高斯黑盒攻击（SEGA）。其主要思想是通过对源模型应用高斯平滑并将其平滑梯度进行合成来近似目标模型的梯度。为了确保对抗性扰动的不可感知性，SEGA使用专门设计的扰动过滤器掩码进一步去除不适当的扰动。CLIVE数据集上的实验结果表明，SEGA具有优越的可移植性，验证了其有效性，使成功的转移为基础的黑盒攻击对NR-IQA模型。



## **14. Zero-Shot Visual Deepfake Detection: Can AI Predict and Prevent Fake Content Before It's Created?**

零镜头视觉深度造假检测：人工智能能否在虚假内容创建之前预测和预防？ cs.GR

Published in Foundations and Trends in Signal Processing (#1 in  Signal Processing, #3 in Computer Science)

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.18461v1) [paper-pdf](http://arxiv.org/pdf/2509.18461v1)

**Authors**: Ayan Sar, Sampurna Roy, Tanupriya Choudhury, Ajith Abraham

**Abstract**: Generative adversarial networks (GANs) and diffusion models have dramatically advanced deepfake technology, and its threats to digital security, media integrity, and public trust have increased rapidly. This research explored zero-shot deepfake detection, an emerging method even when the models have never seen a particular deepfake variation. In this work, we studied self-supervised learning, transformer-based zero-shot classifier, generative model fingerprinting, and meta-learning techniques that better adapt to the ever-evolving deepfake threat. In addition, we suggested AI-driven prevention strategies that mitigated the underlying generation pipeline of the deepfakes before they occurred. They consisted of adversarial perturbations for creating deepfake generators, digital watermarking for content authenticity verification, real-time AI monitoring for content creation pipelines, and blockchain-based content verification frameworks. Despite these advancements, zero-shot detection and prevention faced critical challenges such as adversarial attacks, scalability constraints, ethical dilemmas, and the absence of standardized evaluation benchmarks. These limitations were addressed by discussing future research directions on explainable AI for deepfake detection, multimodal fusion based on image, audio, and text analysis, quantum AI for enhanced security, and federated learning for privacy-preserving deepfake detection. This further highlighted the need for an integrated defense framework for digital authenticity that utilized zero-shot learning in combination with preventive deepfake mechanisms. Finally, we highlighted the important role of interdisciplinary collaboration between AI researchers, cybersecurity experts, and policymakers to create resilient defenses against the rising tide of deepfake attacks.

摘要: 生成对抗网络（GAN）和扩散模型极大地推进了深度伪造技术，其对数字安全、媒体完整性和公众信任的威胁迅速增加。这项研究探索了零镜头深度伪造检测，这是一种新兴方法，即使模型从未见过特定的深度伪造变体。在这项工作中，我们研究了自我监督学习、基于变换器的零次分类器、生成模型指纹识别和元学习技术，这些技术更好地适应不断发展的Deepfake威胁。此外，我们还提出了人工智能驱动的预防策略，可以在Deepfakes发生之前缓解其潜在的生成渠道。它们包括用于创建Deepfake生成器的对抗性扰动、用于内容真实性验证的数字水印、用于内容创建管道的实时人工智能监控以及基于区块链的内容验证框架。尽管取得了这些进步，零攻击检测和预防仍面临着严峻的挑战，例如对抗性攻击、可扩展性限制、道德困境以及缺乏标准化评估基准。通过讨论用于Deepfake检测的可解释人工智能、基于图像、音频和文本分析的多模式融合、用于增强安全性的量子人工智能以及用于保护隐私的Deepfake检测的联邦学习的未来研究方向来解决这些限制。这进一步凸显了对数字真实性集成防御框架的需求，该框架利用零射击学习与预防性深度伪造机制相结合。最后，我们强调了人工智能研究人员、网络安全专家和政策制定者之间跨学科合作的重要作用，以创建针对不断上升的深度伪造攻击浪潮的弹性防御。



## **15. VoxGuard: Evaluating User and Attribute Privacy in Speech via Membership Inference Attacks**

VoxGuard：通过成员资格推理攻击评估语音中的用户和属性隐私 cs.CR

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.18413v1) [paper-pdf](http://arxiv.org/pdf/2509.18413v1)

**Authors**: Efthymios Tsaprazlis, Thanathai Lertpetchpun, Tiantian Feng, Sai Praneeth Karimireddy, Shrikanth Narayanan

**Abstract**: Voice anonymization aims to conceal speaker identity and attributes while preserving intelligibility, but current evaluations rely almost exclusively on Equal Error Rate (EER) that obscures whether adversaries can mount high-precision attacks. We argue that privacy should instead be evaluated in the low false-positive rate (FPR) regime, where even a small number of successful identifications constitutes a meaningful breach. To this end, we introduce VoxGuard, a framework grounded in differential privacy and membership inference that formalizes two complementary notions: User Privacy, preventing speaker re-identification, and Attribute Privacy, protecting sensitive traits such as gender and accent. Across synthetic and real datasets, we find that informed adversaries, especially those using fine-tuned models and max-similarity scoring, achieve orders-of-magnitude stronger attacks at low-FPR despite similar EER. For attributes, we show that simple transparent attacks recover gender and accent with near-perfect accuracy even after anonymization. Our results demonstrate that EER substantially underestimates leakage, highlighting the need for low-FPR evaluation, and recommend VoxGuard as a benchmark for evaluating privacy leakage.

摘要: 语音匿名化旨在隐藏说话者身份和属性，同时保持清晰度，但当前的评估几乎完全依赖于等错误率（EER），该等错误率掩盖了对手是否可以发起高精度攻击。我们认为，隐私应该在低假阳性率（FPR）制度下进行评估，在这种制度下，即使是少数成功的身份识别也构成有意义的侵犯。为此，我们引入了VoxGuard，这是一个基于差异隐私和成员资格推断的框架，它正式化了两个补充概念：用户隐私（防止说话者重新识别）和属性隐私（保护性别和口音等敏感特征）。在合成和真实数据集中，我们发现，尽管EER相似，但知情的对手，尤其是那些使用微调模型和最大相似性评分的对手，在低FPR下实现了数量级的更强攻击。对于属性，我们表明，即使在匿名化之后，简单的透明攻击也能以近乎完美的准确性恢复性别和口音。我们的结果表明，EER大大低估了泄露，强调了低FPR评估的必要性，并建议将VoxGuard作为评估隐私泄露的基准。



## **16. Hybrid Reputation Aggregation: A Robust Defense Mechanism for Adversarial Federated Learning in 5G and Edge Network Environments**

混合声誉聚合：5G和边缘网络环境中对抗联邦学习的稳健防御机制 cs.CR

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.18044v1) [paper-pdf](http://arxiv.org/pdf/2509.18044v1)

**Authors**: Saeid Sheikhi, Panos Kostakos, Lauri Loven

**Abstract**: Federated Learning (FL) in 5G and edge network environments face severe security threats from adversarial clients. Malicious participants can perform label flipping, inject backdoor triggers, or launch Sybil attacks to corrupt the global model. This paper introduces Hybrid Reputation Aggregation (HRA), a novel robust aggregation mechanism designed to defend against diverse adversarial behaviors in FL without prior knowledge of the attack type. HRA combines geometric anomaly detection with momentum-based reputation tracking of clients. In each round, it detects outlier model updates via distance-based geometric analysis while continuously updating a trust score for each client based on historical behavior. This hybrid approach enables adaptive filtering of suspicious updates and long-term penalization of unreliable clients, countering attacks ranging from backdoor insertions to random noise Byzantine failures. We evaluate HRA on a large-scale proprietary 5G network dataset (3M+ records) and the widely used NF-CSE-CIC-IDS2018 benchmark under diverse adversarial attack scenarios. Experimental results reveal that HRA achieves robust global model accuracy of up to 98.66% on the 5G dataset and 96.60% on NF-CSE-CIC-IDS2018, outperforming state-of-the-art aggregators such as Krum, Trimmed Mean, and Bulyan by significant margins. Our ablation studies further demonstrate that the full hybrid system achieves 98.66% accuracy, while the anomaly-only and reputation-only variants drop to 84.77% and 78.52%, respectively, validating the synergistic value of our dual-mechanism approach. This demonstrates HRA's enhanced resilience and robustness in 5G/edge federated learning deployments, even under significant adversarial conditions.

摘要: 5G和边缘网络环境中的联邦学习（FL）面临来自敌对客户端的严重安全威胁。恶意参与者可以执行标签翻转、注入后门触发器或发起Sybil攻击以破坏全球模型。本文介绍了混合声誉聚合（HRA），这是一种新型的鲁棒聚合机制，旨在在不了解攻击类型的情况下防御FL中的各种对抗行为。HRA将几何异常检测与基于动量的客户声誉跟踪相结合。在每一轮中，它通过基于距离的几何分析来检测异常值模型更新，同时根据历史行为不断更新每个客户端的信任分数。这种混合方法能够对可疑更新进行自适应过滤，并对不可靠的客户端进行长期惩罚，抵御从后门插入到随机噪音拜占庭失败等攻击。我们在各种对抗性攻击场景下对大规模专有5G网络数据集（3 M+记录）和广泛使用的NF-CSE-CIC-IDS 2018基准进行评估。实验结果表明，HRA在5G数据集上实现了高达98.66%的稳健全球模型准确率，在NF-CSE-CIC-IDS 2018上实现了高达96.60%的稳健全球模型准确率，远远超过了Krum、Trimmed Mean和Bulyan等最先进的聚合器。我们的消融研究进一步证明，全混合系统的准确率达到了98.66%，而仅异常和仅声誉变体的准确率分别下降到84.77%和78.52%，验证了我们双机制方法的协同价值。这表明HRA在5G/边缘联合学习部署中的弹性和稳健性增强，即使在严重的对抗条件下也是如此。



## **17. Budgeted Adversarial Attack against Graph-Based Anomaly Detection in Sensor Networks**

针对传感器网络中基于图的异常检测的潜在对抗攻击 cs.LG

12 pages

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17987v1) [paper-pdf](http://arxiv.org/pdf/2509.17987v1)

**Authors**: Sanju Xaviar, Omid Ardakanian

**Abstract**: Graph Neural Networks (GNNs) have emerged as powerful models for anomaly detection in sensor networks, particularly when analyzing multivariate time series. In this work, we introduce BETA, a novel grey-box evasion attack targeting such GNN-based detectors, where the attacker is constrained to perturb sensor readings from a limited set of nodes, excluding the target sensor, with the goal of either suppressing a true anomaly or triggering a false alarm at the target node. BETA identifies the sensors most influential to the target node's classification and injects carefully crafted adversarial perturbations into their features, all while maintaining stealth and respecting the attacker's budget. Experiments on three real-world sensor network datasets show that BETA reduces the detection accuracy of state-of-the-art GNN-based detectors by 30.62 to 39.16% on average, and significantly outperforms baseline attack strategies, while operating within realistic constraints.

摘要: 图神经网络（GNN）已成为传感器网络异常检测的强大模型，特别是在分析多元时间序列时。在这项工作中，我们引入了BETA，这是一种针对此类基于GNN的检测器的新型灰箱规避攻击，其中攻击者被限制为干扰来自有限节点集（不包括目标传感器）的传感器读数，目标是要么抑制真实异常，要么在目标节点触发虚警。Beta识别对目标节点分类最有影响的传感器，并将精心设计的对抗性扰动注入到其特征中，同时保持隐形并尊重攻击者的预算。对三个现实世界传感器网络数据集的实验表明，BEP将最先进的基于GNN的检测器的检测准确性平均降低了30.62%至39.16%，并且显着优于基线攻击策略，同时在现实的限制范围内运行。



## **18. A Lightweight Approach for State Machine Replication**

状态机复制的轻量级方法 cs.DC

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17771v1) [paper-pdf](http://arxiv.org/pdf/2509.17771v1)

**Authors**: Christian Cachin, Jinfeng Dou, Christian Scheideler, Philipp Schneider

**Abstract**: We present a lightweight solution for state machine replication with commitment certificates. Specifically, we adapt a simple median rule from the stabilizing consensus problem [Doerr11] to operate in a client-server setting where arbitrary servers may be blocked adaptively based on past system information. We further extend our protocol by compressing information about committed commands, thus keeping the protocol lightweight, while still enabling clients to easily prove that their commands have indeed been committed on the shared state. Our approach guarantees liveness as long as at most a constant fraction of servers are blocked, ensures safety under any number of blocked servers, and supports fast recovery from massive blocking attacks. In addition to offering near-optimal performance in several respects, our method is fully decentralized, unlike other near-optimal solutions that rely on leaders. In particular, our solution is robust against adversaries that target key servers (which captures insider-based denial-of-service attacks), whereas leader-based approaches fail under such a blocking model.

摘要: 我们提供了一种用于具有承诺证书的状态机复制的轻量级解决方案。具体来说，我们根据稳定共识问题[Doerr 11]调整了一个简单的中位数规则，以在客户端-服务器设置中操作，其中可以根据过去的系统信息自适应地阻止任意服务器。我们通过压缩有关已提交命令的信息来进一步扩展我们的协议，从而保持协议轻量级，同时仍然使客户端能够轻松证明他们的命令确实已在共享状态上提交。只要最多有一定比例的服务器被阻止，我们的方法就能保证活跃性，确保任何数量被阻止的服务器下的安全性，并支持从大规模阻止攻击中快速恢复。除了在几个方面提供近乎最优的性能外，我们的方法还完全去中心化，与其他依赖领导者的近乎最优的解决方案不同。特别是，我们的解决方案对于针对关键服务器（捕获基于内部的拒绝服务攻击）的对手来说是强大的，而基于领导者的方法在这种阻止模型下会失败。



## **19. Is It Certainly a Deepfake? Reliability Analysis in Detection & Generation Ecosystem**

难道真的是假的吗？检测与发电生态系统中的可靠性分析 cs.AI

Accepted for publication at the ICCV 2025 STREAM workshop

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17550v1) [paper-pdf](http://arxiv.org/pdf/2509.17550v1)

**Authors**: Neslihan Kose, Anthony Rhodes, Umur Aybars Ciftci, Ilke Demir

**Abstract**: As generative models are advancing in quality and quantity for creating synthetic content, deepfakes begin to cause online mistrust. Deepfake detectors are proposed to counter this effect, however, misuse of detectors claiming fake content as real or vice versa further fuels this misinformation problem. We present the first comprehensive uncertainty analysis of deepfake detectors, systematically investigating how generative artifacts influence prediction confidence. As reflected in detectors' responses, deepfake generators also contribute to this uncertainty as their generative residues vary, so we cross the uncertainty analysis of deepfake detectors and generators. Based on our observations, the uncertainty manifold holds enough consistent information to leverage uncertainty for deepfake source detection. Our approach leverages Bayesian Neural Networks and Monte Carlo dropout to quantify both aleatoric and epistemic uncertainties across diverse detector architectures. We evaluate uncertainty on two datasets with nine generators, with four blind and two biological detectors, compare different uncertainty methods, explore region- and pixel-based uncertainty, and conduct ablation studies. We conduct and analyze binary real/fake, multi-class real/fake, source detection, and leave-one-out experiments between the generator/detector combinations to share their generalization capability, model calibration, uncertainty, and robustness against adversarial attacks. We further introduce uncertainty maps that localize prediction confidence at the pixel level, revealing distinct patterns correlated with generator-specific artifacts. Our analysis provides critical insights for deploying reliable deepfake detection systems and establishes uncertainty quantification as a fundamental requirement for trustworthy synthetic media detection.

摘要: 随着生成模型在创建合成内容的质量和数量上不断进步，深度造假开始引起在线不信任。Deepfake检测器被提出来对抗这种影响，然而，滥用检测器将虚假内容称为真实内容或反之亦然，进一步加剧了这种错误信息问题。我们首次对Deepfake检测器进行了全面的不确定性分析，系统地研究生成伪影如何影响预测置信度。正如检测器的响应所反映的那样，Deepfake生成器也会导致这种不确定性，因为它们的生成残余有所不同，因此我们交叉了Deepfake检测器和生成器的不确定性分析。根据我们的观察，不确定性集合包含足够一致的信息，可以利用不确定性进行深度伪造源检测。我们的方法利用Bayesian神经网络和Monte Carlo dropout来量化不同检测器架构中的任意性和认识性不确定性。我们评估了两个数据集的不确定性，该数据集具有九个发生器、四个盲探测器和两个生物探测器，比较不同的不确定性方法，探索基于区域和像素的不确定性，并进行消融研究。我们在生成器/检测器组合之间进行并分析二进制实/假、多类实/假、源检测和留一实验，以分享它们的概括能力、模型校准、不确定性和对抗性攻击的鲁棒性。我们进一步引入不确定性地图，在像素级定位预测的信心，揭示了不同的模式与发电机特定的文物。我们的分析为部署可靠的deepfake检测系统提供了重要见解，并将不确定性量化确定为可信合成媒体检测的基本要求。



## **20. Proxy-Embedding as an Adversarial Teacher: An Embedding-Guided Bidirectional Attack for Referring Expression Segmentation Models**

代理嵌入作为对抗性教师：引用表情分割模型的嵌入引导双向攻击 cs.CV

20pages, 5figures

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2506.16157v2) [paper-pdf](http://arxiv.org/pdf/2506.16157v2)

**Authors**: Xingbai Chen, Tingchao Fu, Renyang Liu, Wei Zhou, Chao Yi

**Abstract**: Referring Expression Segmentation (RES) enables precise object segmentation in images based on natural language descriptions, offering high flexibility and broad applicability in real-world vision tasks. Despite its impressive performance, the robustness of RES models against adversarial examples remains largely unexplored. While prior adversarial attack methods have explored adversarial robustness on conventional segmentation models, they perform poorly when directly applied to RES models, failing to expose vulnerabilities in its multimodal structure. In practical open-world scenarios, users typically issue multiple, diverse referring expressions to interact with the same image, highlighting the need for adversarial examples that generalize across varied textual inputs. Furthermore, from the perspective of privacy protection, ensuring that RES models do not segment sensitive content without explicit authorization is a crucial aspect of enhancing the robustness and security of multimodal vision-language systems. To address these challenges, we present PEAT, an Embedding-Guided Bidirectional Attack for RES models. Extensive experiments across multiple RES architectures and standard benchmarks show that PEAT consistently outperforms competitive baselines.

摘要: 引用表达分割（RES）可以基于自然语言描述在图像中进行精确的对象分割，在现实世界的视觉任务中提供高度灵活性和广泛的适用性。尽管RES模型的性能令人印象深刻，但其针对对抗性示例的稳健性在很大程度上仍未得到探索。虽然先前的对抗攻击方法已经探索了传统分割模型的对抗鲁棒性，但当直接应用于RES模型时，它们的表现很差，未能暴露其多模式结构中的漏洞。在实际的开放世界场景中，用户通常会发出多个不同的引用表达来与同一图像进行交互，这凸显了对跨越不同文本输入进行概括的对抗性示例的需要。此外，从隐私保护的角度来看，确保RES模型在没有明确授权的情况下不会分割敏感内容，是增强多模态视觉语言系统的鲁棒性和安全性的关键方面。为了解决这些挑战，我们提出了PEAT，一种针对RES模型的嵌入引导双向攻击。跨多个RES架构和标准基准的广泛实验表明，PEAT始终优于竞争对手的基线。



## **21. Explainable AI for Analyzing Person-Specific Patterns in Facial Recognition Tasks**

用于分析面部识别任务中特定于个人的模式的可解释人工智能 cs.CV

22 pages; 24 tables; 11 figures

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17457v1) [paper-pdf](http://arxiv.org/pdf/2509.17457v1)

**Authors**: Paweł Jakub Borsukiewicz, Jordan Samhi, Jacques Klein, Tegawendé F. Bissyandé

**Abstract**: The proliferation of facial recognition systems presents major privacy risks, driving the need for effective countermeasures. Current adversarial techniques apply generalized methods rather than adapting to individual facial characteristics, limiting their effectiveness and inconspicuousness. In this work, we introduce Layer Embedding Activation Mapping (LEAM), a novel technique that identifies which facial areas contribute most to recognition at an individual level. Unlike adversarial attack methods that aim to fool recognition systems, LEAM is an explainability technique designed to understand how these systems work, providing insights that could inform future privacy protection research. We integrate LEAM with a face parser to analyze data from 1000 individuals across 9 pre-trained facial recognition models.   Our analysis reveals that while different layers within facial recognition models vary significantly in their focus areas, these models generally prioritize similar facial regions across architectures when considering their overall activation patterns, which show significantly higher similarity between images of the same individual (Bhattacharyya Coefficient: 0.32-0.57) vs. different individuals (0.04-0.13), validating the existence of person-specific recognition patterns. Our results show that facial recognition models prioritize the central region of face images (with nose areas accounting for 18.9-29.7% of critical recognition regions), while still distributing attention across multiple facial fragments. Proper selection of relevant facial areas was confirmed using validation occlusions, based on just 1% of the most relevant, LEAM-identified, image pixels, which proved to be transferable across different models. Our findings establish the foundation for future individually tailored privacy protection systems centered around LEAM's choice of areas to be perturbed.

摘要: 面部识别系统的激增带来了重大的隐私风险，促使人们需要有效的应对措施。当前的对抗技术应用一般化的方法，而不是适应个人的面部特征，从而限制了其有效性和不引人注目性。在这项工作中，我们引入了层嵌入激活映射（LEAM），这是一种新颖的技术，可以识别哪些面部区域对个人层面的识别贡献最大。与旨在欺骗识别系统的对抗性攻击方法不同，LEAM是一种可解释性技术，旨在了解这些系统如何工作，提供可以为未来隐私保护研究提供信息的见解。我们将LEAM与面部解析器集成，以分析来自1000个人的9个预训练面部识别模型的数据。   我们的分析表明，虽然面部识别模型中的不同层在其焦点区域上存在显着差异，但这些模型在考虑其整体激活模式时通常会优先考虑跨架构的相似面部区域，这表明同一个人的图像之间的相似性显着更高（Bhattacharyya系数：0.32-0.57）与不同个体（0.04-0.13），验证了特定于个体的识别模式的存在。我们的结果表明，面部识别模型优先考虑面部图像的中心区域（鼻子区域占关键识别区域的18.9-29.7%），同时仍然将注意力分散在多个面部片段上。仅基于1%的最相关的LEAM识别图像像素，使用验证遮挡来确认相关面部区域的正确选择，事实证明这些像素可以在不同的模型之间转移。我们的研究结果为未来个性化定制的隐私保护系统奠定了基础，该系统以LEAM选择的受干扰区域为中心。



## **22. TextCrafter: Optimization-Calibrated Noise for Defending Against Text Embedding Inversion**

TextCrafter：用于防御文本嵌入倒置的优化校准噪音 cs.CR

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.17302v1) [paper-pdf](http://arxiv.org/pdf/2509.17302v1)

**Authors**: Duoxun Tang, Xinhang Jiang, Jiajun Niu

**Abstract**: Text embedding inversion attacks reconstruct original sentences from latent representations, posing severe privacy threats in collaborative inference and edge computing. We propose TextCrafter, an optimization-based adversarial perturbation mechanism that combines RL learned, geometry aware noise injection orthogonal to user embeddings with cluster priors and PII signal guidance to suppress inversion while preserving task utility. Unlike prior defenses either non learnable or agnostic to perturbation direction, TextCrafter provides a directional protective policy that balances privacy and utility. Under strong privacy setting, TextCrafter maintains 70 percentage classification accuracy on four datasets and consistently outperforms Gaussian/LDP baselines across lower privacy budgets, demonstrating a superior privacy utility trade off.

摘要: 文本嵌入倒置攻击从潜在表示中重建原始句子，在协作推理和边缘计算中构成严重的隐私威胁。我们提出了TextCrafter，这是一种基于优化的对抗性扰动机制，它将RL学习的、与用户嵌入垂直的几何感知噪音注入与集群先验和PRI信号引导相结合，以抑制倒置，同时保留任务效用。与之前的防御（无论是不可学习的还是对扰动方向不可知的）不同，TextCrafter提供了一种平衡隐私和实用性的定向保护政策。在强大的隐私设置下，TextCrafter在四个数据集上保持了70%的分类准确性，并且在较低的隐私预算下始终优于高斯/LDP基线，展示了卓越的隐私实用权衡。



## **23. Seeing is Deceiving: Mirror-Based LiDAR Spoofing for Autonomous Vehicle Deception**

亲眼所见：基于地理位置的激光雷达欺骗自动驾驶车辆 cs.CR

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17253v1) [paper-pdf](http://arxiv.org/pdf/2509.17253v1)

**Authors**: Selma Yahia, Ildi Alla, Girija Bangalore Mohan, Daniel Rau, Mridula Singh, Valeria Loscri

**Abstract**: Autonomous vehicles (AVs) rely heavily on LiDAR sensors for accurate 3D perception. We show a novel class of low-cost, passive LiDAR spoofing attacks that exploit mirror-like surfaces to inject or remove objects from an AV's perception. Using planar mirrors to redirect LiDAR beams, these attacks require no electronics or custom fabrication and can be deployed in real settings. We define two adversarial goals: Object Addition Attacks (OAA), which create phantom obstacles, and Object Removal Attacks (ORA), which conceal real hazards. We develop geometric optics models, validate them with controlled outdoor experiments using a commercial LiDAR and an Autoware-equipped vehicle, and implement a CARLA-based simulation for scalable testing. Experiments show mirror attacks corrupt occupancy grids, induce false detections, and trigger unsafe planning and control behaviors. We discuss potential defenses (thermal sensing, multi-sensor fusion, light-fingerprinting) and their limitations.

摘要: 自动驾驶汽车（AV）严重依赖LiDART传感器来实现准确的3D感知。我们展示了一类新型的低成本、被动的LiDART欺骗攻击，它们利用类似镜子的表面来从AV的感知中注入或删除对象。这些攻击使用平面镜来重定向LiDART射束，不需要电子设备或定制制造，并且可以部署在真实环境中。我们定义了两个对抗目标：创建幻影障碍的对象添加攻击（OAA）和隐藏真正危险的对象删除攻击（ORA）。我们开发几何光学模型，使用商用LiDART和配备自动软件的车辆通过受控户外实验对其进行验证，并实施基于CARLA的模拟以进行可扩展测试。实验表明，镜像攻击会破坏占用网格，引发错误检测，并引发不安全的规划和控制行为。我们讨论了潜在的防御（热传感、多传感器融合、光指纹识别）及其局限性。



## **24. TraceHiding: Scalable Machine Unlearning for Mobility Data**

TraceHiding：移动数据的可扩展机器去学习 cs.LG

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17241v1) [paper-pdf](http://arxiv.org/pdf/2509.17241v1)

**Authors**: Ali Faraji, Manos Papagelis

**Abstract**: This work introduces TraceHiding, a scalable, importance-aware machine unlearning framework for mobility trajectory data. Motivated by privacy regulations such as GDPR and CCPA granting users "the right to be forgotten," TraceHiding removes specified user trajectories from trained deep models without full retraining. It combines a hierarchical data-driven importance scoring scheme with teacher-student distillation. Importance scores--computed at token, trajectory, and user levels from statistical properties (coverage diversity, entropy, length)--quantify each training sample's impact, enabling targeted forgetting of high-impact data while preserving common patterns. The student model retains knowledge on remaining data and unlearns targeted trajectories through an importance-weighted loss that amplifies forgetting signals for unique samples and attenuates them for frequent ones. We validate on Trajectory--User Linking (TUL) tasks across three real-world higher-order mobility datasets (HO-Rome, HO-Geolife, HO-NYC) and multiple architectures (GRU, LSTM, BERT, ModernBERT, GCN-TULHOR), against strong unlearning baselines including SCRUB, NegGrad, NegGrad+, Bad-T, and Finetuning. Experiments under uniform and targeted user deletion show TraceHiding, especially its entropy-based variant, achieves superior unlearning accuracy, competitive membership inference attack (MIA) resilience, and up to 40\times speedup over retraining with minimal test accuracy loss. Results highlight robustness to adversarial deletion of high-information users and consistent performance across models. To our knowledge, this is the first systematic study of machine unlearning for trajectory data, providing a reproducible pipeline with public code and preprocessing tools.

摘要: 这项工作引入了TraceHiding，这是一个针对移动轨迹数据的可扩展、重要性感知的机器去学习框架。受GDPR和CCPA等隐私法规的激励，TraceHiding赋予用户“被遗忘的权利”，无需进行全面重新培训，即可从训练后的深度模型中删除指定的用户轨迹。它将分层数据驱动的重要性评分方案与师生蒸馏相结合。重要性分数--根据统计属性（覆盖多样性、信息量、长度）在代币、轨迹和用户级别上计算--量化每个训练样本的影响，从而能够有针对性地忘记高影响力数据，同时保留共同模式。学生模型保留了剩余数据的知识，并通过重要性加权损失来忘记目标轨迹，该损失放大了独特样本的遗忘信号，并削弱了频繁样本的遗忘信号。我们在三个现实世界的更高级移动性数据集（HO-Rome、HO-Geolife、HO-NYC）和多个架构（GRU、LSTM、BERT、ModernBERT、GCN-TULHOR）上验证了Trajectory--用户链接（TUL）任务，针对强大的取消学习基线（包括SCRUB、NegGrad、NegGrad+、Bad-T和Finetuning）。在统一和有针对性的用户删除下的实验表明，TraceHiding，尤其是其基于信息量的变体，实现了卓越的取消学习准确性、竞争性成员推断攻击（MIA）弹性，以及比再训练高达40倍的加速，测试准确性损失最小。结果凸显了高信息用户对抗删除的稳健性以及模型之间一致的性能。据我们所知，这是第一次对轨迹数据的机器去学习进行系统研究，提供了具有公共代码和预处理工具的可重复管道。



## **25. The Cost of Compression: Tight Quadratic Black-Box Attacks on Sketches for $\ell_2$ Norm Estimation**

压缩的成本：对草图进行严格的二次黑匣子攻击，以获取$\ell_2 $ Norm估计 cs.LG

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2507.16345v2) [paper-pdf](http://arxiv.org/pdf/2507.16345v2)

**Authors**: Sara Ahmadian, Edith Cohen, Uri Stemmer

**Abstract**: Dimensionality reduction via linear sketching is a powerful and widely used technique, but it is known to be vulnerable to adversarial inputs. We study the black-box adversarial setting, where a fixed, hidden sketching matrix $A \in R^{k \times n}$ maps high-dimensional vectors $v \in R^n$ to lower-dimensional sketches $A v \in R^k$, and an adversary can query the system to obtain approximate $\ell_2$-norm estimates that are computed from the sketch. We present a universal, nonadaptive attack that, using $\tilde{O}(k^2)$ queries, either causes a failure in norm estimation or constructs an adversarial input on which the optimal estimator for the query distribution (used by the attack) fails. The attack is completely agnostic to the sketching matrix and to the estimator: it applies to any linear sketch and any query responder, including those that are randomized, adaptive, or tailored to the query distribution. Our lower bound construction tightly matches the known upper bounds of $\tilde{\Omega}(k^2)$, achieved by specialized estimators for Johnson Lindenstrauss transforms and AMS sketches. Beyond sketching, our results uncover structural parallels to adversarial attacks in image classification, highlighting fundamental vulnerabilities of compressed representations.

摘要: 通过线性草图减少维度是一种强大且广泛使用的技术，但众所周知，它容易受到对抗输入的影响。我们研究黑匣子对抗设置，其中固定的、隐藏的草图矩阵$A \in R^{k \times n}$将高维载体$v \in R^n$映射到低维草图$A v \in R^k$，对手可以查询系统以获得从草图计算的大约$\ell_2 $-norm估计。我们提出了一种通用的非适应性攻击，使用$\tilde{O}（k#2）$查询，要么导致规范估计失败，要么构建对抗输入，查询分布的最佳估计器（由攻击使用）失败。该攻击对草图矩阵和估计器完全不可知：它适用于任何线性草图和任何查询响应器，包括随机化、自适应或针对查询分布定制的那些。我们的下限结构与$\tilde{\Omega}（k#2）$的已知上限紧密匹配，该上限由Johnson Lindenstrauss变换和AMS草图的专门估计器实现。除了草图之外，我们的结果还揭示了图像分类中与对抗攻击的结构相似之处，凸显了压缩表示的基本漏洞。



## **26. Bribers, Bribers on The Chain, Is Resisting All in Vain? Trustless Consensus Manipulation Through Bribing Contracts**

行贿者，链上的行贿者，抵制一切都是徒劳的吗？通过贿赂合同进行不可信的共识操纵 cs.CR

pre-print

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17185v1) [paper-pdf](http://arxiv.org/pdf/2509.17185v1)

**Authors**: Bence Soóki-Tóth, István András Seres, Kamilla Kara, Ábel Nagy, Balázs Pejó, Gergely Biczók

**Abstract**: The long-term success of cryptocurrencies largely depends on the incentive compatibility provided to the validators. Bribery attacks, facilitated trustlessly via smart contracts, threaten this foundation. This work introduces, implements, and evaluates three novel and efficient bribery contracts targeting Ethereum validators. The first bribery contract enables a briber to fork the blockchain by buying votes on their proposed blocks. The second contract incentivizes validators to voluntarily exit the consensus protocol, thus increasing the adversary's relative staking power. The third contract builds a trustless bribery market that enables the briber to auction off their manipulative power over the RANDAO, Ethereum's distributed randomness beacon. Finally, we provide an initial game-theoretical analysis of one of the described bribery markets.

摘要: 加密货币的长期成功很大程度上取决于为验证者提供的激励兼容性。通过智能合同不受信任地促进的贿赂攻击威胁着这一基础。这项工作介绍、实施和评估了三个针对以太坊验证者的新颖且高效的贿赂合同。第一份贿赂合同使贿赂者能够通过购买其提议区块的选票来分叉区块链。第二个合同激励验证者自愿退出共识协议，从而增加对手的相对赌注权力。第三个合同建立了一个不可信任的贿赂市场，使贿赂者能够拍卖他们对以太坊的分布式随机性信标RANDO的操纵权力。最后，我们对所描述的贿赂市场之一进行了初步的博弈论分析。



## **27. Unaligned Incentives: Pricing Attacks Against Blockchain Rollups**

不一致的激励措施：针对区块链汇总的定价攻击 cs.CR

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17126v1) [paper-pdf](http://arxiv.org/pdf/2509.17126v1)

**Authors**: Stefanos Chaliasos, Conner Swann, Sina Pilehchiha, Nicolas Mohnblatt, Benjamin Livshits, Assimakis Kattis

**Abstract**: Rollups have become the de facto scalability solution for Ethereum, securing more than $55B in assets. They achieve scale by executing transactions on a Layer 2 ledger, while periodically posting data and finalizing state on the Layer 1, either optimistically or via validity proofs. Their fees must simultaneously reflect the pricing of three resources: L2 costs (e.g., execution), L1 DA, and underlying L1 gas costs for batch settlement and proof verification. In this work, we identify critical mis-pricings in existing rollup transaction fee mechanisms (TFMs) that allow for two powerful attacks. Firstly, an adversary can saturate the L2's DA batch capacity with compute-light data-heavy transactions, forcing low-gas transaction batches that enable both L2 DoS attacks, and finality-delay attacks. Secondly, by crafting prover killer transactions that maximize proving cycles relative to the gas charges, an adversary can effectively stall proof generation, delaying finality by hours and inflicting prover-side economic losses to the rollup at a minimal cost.   We analyze the above attack vectors across the major Ethereum rollups, quantifying adversarial costs and protocol losses. We find that the first attack enables periodic DoS on rollups, lasting up to 30 minutes, at a cost below 2 ETH for most rollups. Moreover, we identify three rollups that are exposed to indefinite DoS at a cost of approximately 0.8 to 2.7 ETH per hour. The attack can be further modified to increase finalization delays by a factor of about 1.45x to 2.73x, compared to direct L1 blob-stuffing, depending on the rollup's parameters. Furthermore, we find that the prover killer attack induces a finalization latency increase of about 94x. Finally, we propose comprehensive mitigations to prevent these attacks and suggest how some practical uses of multi-dimensional rollup TFMs can rectify the identified mis-pricing attacks.

摘要: Rollups已经成为以太坊事实上的可扩展性解决方案，保护了超过550亿美元的资产。它们通过在第2层分类账上执行交易来实现规模，同时定期发布数据并在第1层上完成状态，无论是乐观地还是通过有效性证明。他们的费用必须同时反映三种资源的定价：L2成本（例如，执行）、L1 DA和基础L1 gas成本，用于批量结算和证明验证。在这项工作中，我们发现了现有汇总交易费用机制（TFM）中的严重错误定价，这些机制允许两种强大的攻击。首先，对手可以通过轻计算、数据量大的事务来饱和L2的DA批处理容量，从而强制进行低气体事务批处理，从而支持L2 DPS攻击和最终延迟攻击。其次，通过精心设计证明者杀手级交易，最大限度地提高相对于天然气费用的证明周期，对手可以有效地阻止证明的生成，将最终结果推迟几个小时，并以最低的成本给汇总造成证明方的经济损失。   我们分析了主要以太坊汇总的上述攻击载体，量化对抗成本和协议损失。我们发现，第一次攻击可以对汇总进行定期拒绝服务，持续时间长达30分钟，大多数汇总的成本低于2 ETH。此外，我们发现了三个受无限期拒绝服务的汇总，费用约为每小时0.8至2.7 ETH。与直接L1斑点填充相比，可以进一步修改攻击，以将最终延迟增加约1.45x至2.73x，具体取决于汇总的参数。此外，我们发现证明者杀手攻击导致最终确定延迟增加约94倍。最后，我们提出了全面的缓解措施来防止这些攻击，并建议如何使用多维汇总TLR来纠正已识别的错误定价攻击。



## **28. SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions**

SVeritas：不同条件下稳健说话人验证的基准 cs.SD

Accepted to EMNLP 2025 Findings

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.17091v1) [paper-pdf](http://arxiv.org/pdf/2509.17091v1)

**Authors**: Massa Baali, Sarthak Bisht, Francisco Teixeira, Kateryna Shapovalenko, Rita Singh, Bhiksha Raj

**Abstract**: Speaker verification (SV) models are increasingly integrated into security, personalization, and access control systems, yet their robustness to many real-world challenges remains inadequately benchmarked. These include a variety of natural and maliciously created conditions causing signal degradations or mismatches between enrollment and test data, impacting performance. Existing benchmarks evaluate only subsets of these conditions, missing others entirely. We introduce SVeritas, a comprehensive Speaker Verification tasks benchmark suite, assessing SV systems under stressors like recording duration, spontaneity, content, noise, microphone distance, reverberation, channel mismatches, audio bandwidth, codecs, speaker age, and susceptibility to spoofing and adversarial attacks. While several benchmarks do exist that each cover some of these issues, SVeritas is the first comprehensive evaluation that not only includes all of these, but also several other entirely new, but nonetheless important, real-life conditions that have not previously been benchmarked. We use SVeritas to evaluate several state-of-the-art SV models and observe that while some architectures maintain stability under common distortions, they suffer substantial performance degradation in scenarios involving cross-language trials, age mismatches, and codec-induced compression. Extending our analysis across demographic subgroups, we further identify disparities in robustness across age groups, gender, and linguistic backgrounds. By standardizing evaluation under realistic and synthetic stress conditions, SVeritas enables precise diagnosis of model weaknesses and establishes a foundation for advancing equitable and reliable speaker verification systems.

摘要: 说话者验证（SV）模型越来越多地集成到安全、个性化和访问控制系统中，但它们对许多现实世界挑战的稳健性仍然没有充分的基准。其中包括各种自然和恶意创建的条件，导致注册和测试数据之间的信号退化或不匹配，从而影响性能。现有的基准仅评估这些条件的子集，而完全忽略了其他条件。我们引入SVeritas，这是一个全面的扬声器验证任务基准套件，在录音持续时间、自发性、内容、噪音、麦克风距离、回响、通道不匹配、音频带宽、编解码器、扬声器年龄以及对欺骗和对抗性攻击的敏感性等压力源下评估SV系统。虽然确实存在几个基准，每个基准都涵盖了其中的一些问题，但SVeritas是第一个全面的评估，不仅包括所有这些问题，而且还包括其他几个全新的，但仍然重要的，以前没有基准的现实生活条件。我们使用SVeritas来评估几个最先进的SV模型，并观察到，虽然一些架构在常见的失真下保持稳定性，但在涉及跨语言试验，年龄不匹配和编解码器引起的压缩的情况下，它们的性能会大幅下降。将我们的分析扩展到人口统计学亚组，我们进一步确定了年龄组，性别和语言背景之间的鲁棒性差异。通过在真实和合成压力条件下对评估进行标准化，SVeritas能够精确诊断模型的弱点，并为推进公平可靠的说话人验证系统奠定基础。



## **29. Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs**

重新审视对LLM的后门攻击：通过无害输入的隐形且实用的毒害框架 cs.CL

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2505.17601v3) [paper-pdf](http://arxiv.org/pdf/2505.17601v3)

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Ke Xu, Han Qiu

**Abstract**: Recent studies have widely investigated backdoor attacks on Large language models (LLMs) by inserting harmful question-answer (QA) pairs into training data to implant triggers. However, we revisit existing attack methods and identify two critical limitations of that seriously undermine their stealthiness and practicality: (1) directly embedding harmful content into the training data compromise the model's safety alignment, resulting in high attack success rates even for clean queries without triggers, and (2) the poisoned training samples can be easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard). To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, the adversary inputs a malicious query with the trigger activated to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Notably, achieving this behavior from clean QA pairs is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment issue, and design a robust and general benign response template for constructing backdoor training data, which yields strong performance. To further enhance attack efficacy, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method effectively injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models. E.g., ASRs of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.

摘要: 最近的研究广泛调查了对大型语言模型（LLM）的后门攻击，方法是在训练数据中插入有害的问答（QA）对以植入触发器。然而，我们重新审视了现有的攻击方法，并发现了严重损害其隐蔽性和实用性的两个关键局限性：（1）直接将有害内容嵌入到训练数据中会损害模型的安全对齐，即使对于没有触发器的干净查询也会导致很高的攻击成功率，以及（2）中毒的训练样本可以很容易地检测和过滤安全对齐的护栏（例如，LLaMAGuard）。为此，我们提出了一种通过完全无害的数据的新型中毒方法。受到自回归LLM中因果推理的启发，我们的目标是仅使用良性QA对，而不是直接将触发器与有害反应联系起来，在触发器与肯定反应之间建立稳健的关联。在推理过程中，对手输入恶意查询，触发器被激活以引出此肯定性前置。然后，LLM根据其语言建模能力完成响应。值得注意的是，从干净的QA对实现这种行为并非易事。我们观察到一个有趣的阻力现象，LLM最初似乎同意，但随后拒绝回答。我们将其归因于浅层对齐问题，并设计一个稳健且通用的良性响应模板来构建后门训练数据，从而产生强大的性能。为了进一步提高攻击功效，我们通过基于梯度的协调优化改进了通用触发器。大量实验表明，即使在检测到强大的护栏模型的情况下，我们的方法也可以有效地将后门注入到各种LLM中，以生成有害内容。例如，根据GPT-4 o判断，LLaMA-3-8B和Qwen-2.5- 7 B的ASB分别为86.67%和85%。



## **30. Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks**

打破评论者：评估文本对抗攻击下自动同行评审中大型语言模型的脆弱性 cs.CL

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2506.11113v2) [paper-pdf](http://arxiv.org/pdf/2506.11113v2)

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.

摘要: 同行评审对于保持学术质量至关重要，但提交量的增加给评审者带来了沉重的负担。大型语言模型（LLM）在此过程中提供了潜在的帮助，但它们对文本对抗攻击的敏感性引发了可靠性问题。本文研究了在存在此类攻击的情况下用作自动审查员的LLM的稳健性。我们重点关注三个关键问题：（1）与人类评审员相比，LLM在生成评审方面的有效性。(2)对抗性攻击对LLM生成的评论的可靠性的影响。(3)LLM为基础的审查的挑战和潜在的缓解策略。我们的评估揭示了重大的漏洞，因为文本操作可能会扭曲LLM评估。我们提供了一个全面的评估LLM性能的自动同行评审，并分析其对抗攻击的鲁棒性。我们的研究结果强调了解决对抗风险的重要性，以确保人工智能加强而不是损害学术交流的完整性。



## **31. BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability**

BlockA2A：迈向安全且可验证的代理对代理互操作性 cs.CR

43 pages

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2508.01332v3) [paper-pdf](http://arxiv.org/pdf/2508.01332v3)

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments.

摘要: 由大型语言模型（LLM）支持的代理人工智能的快速采用正在通过执行复杂工作流程的自主代理改变企业生态系统。然而，我们在LLM驱动的多代理系统（MASes）中观察到了几个关键的安全漏洞：碎片化的身份框架、不安全的通信渠道以及对拜占庭代理或对抗提示的防御不足。在本文中，我们对这些新出现的多代理风险进行了首次系统分析，并解释了为什么传统安全策略无法有效应对这些风险。随后，我们提出了BlockA2A，这是第一个统一的多代理信任框架，可以实现安全、可验证以及代理与代理的互操作性。在高层面上，BlockA2A采用去中心化标识符（DID）来实现细粒度的跨域代理认证，采用区块链锚定分类帐来实现不可变的可互换性，并采用智能合同来动态执行上下文感知的访问控制策略。BlockA2A消除了集中式信任瓶颈，确保消息真实性和执行完整性，并保证跨代理交互的问责制。此外，我们还提出了一种国防规划引擎（DOE），它通过实时机制主动中和攻击，包括拜占庭代理标记、反应式执行停止和即时许可撤销。经验评估证明BlockA2A在中和基于预算、基于通信的行为和系统性MAS攻击方面的有效性。我们将其正式集成到现有MAS中，并展示了Google A2 A协议的实际实现。实验证实BlockA2A和DOE的运行成本为亚秒级，从而能够在基于LLM的生产MAS环境中进行可扩展部署。



## **32. Dynamical Low-Rank Compression of Neural Networks with Robustness under Adversarial Attacks**

抗攻击鲁棒性神经网络动态低秩压缩 cs.LG

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2505.08022v3) [paper-pdf](http://arxiv.org/pdf/2505.08022v3)

**Authors**: Steffen Schotthöfer, H. Lexie Yang, Stefan Schnake

**Abstract**: Deployment of neural networks on resource-constrained devices demands models that are both compact and robust to adversarial inputs. However, compression and adversarial robustness often conflict. In this work, we introduce a dynamical low-rank training scheme enhanced with a novel spectral regularizer that controls the condition number of the low-rank core in each layer. This approach mitigates the sensitivity of compressed models to adversarial perturbations without sacrificing accuracy on clean data. The method is model- and data-agnostic, computationally efficient, and supports rank adaptivity to automatically compress the network at hand. Extensive experiments across standard architectures, datasets, and adversarial attacks show the regularized networks can achieve over 94% compression while recovering or improving adversarial accuracy relative to uncompressed baselines.

摘要: 在资源受限的设备上部署神经网络需要紧凑且对对抗输入稳健的模型。然而，压缩和对抗鲁棒性经常发生冲突。在这项工作中，我们引入了一种动态低等级训练方案，该方案通过新型谱正规化器增强，该算法控制每层中低等级核心的条件数。这种方法降低了压缩模型对对抗性扰动的敏感性，而不会牺牲干净数据的准确性。该方法与模型和数据无关，计算效率高，并且支持等级自适应性以自动压缩手头的网络。跨标准架构、数据集和对抗性攻击的广泛实验表明，正规化网络可以实现超过94%的压缩，同时恢复或提高相对于未压缩基线的对抗性准确性。



## **33. Securing the Language of Life: Inheritable Watermarks from DNA Language Models to Proteins**

保护生命的语言：从DNA语言模型到蛋白质的可继承水印 q-bio.GN

Accepted by NeurIPS 2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.18207v1) [paper-pdf](http://arxiv.org/pdf/2509.18207v1)

**Authors**: Zaixi Zhang, Ruofan Jin, Le Cong, Mengdi Wang

**Abstract**: DNA language models have revolutionized our ability to understand and design DNA sequences--the fundamental language of life--with unprecedented precision, enabling transformative applications in therapeutics, synthetic biology, and gene editing. However, this capability also poses substantial dual-use risks, including the potential for creating pathogens, viruses, and even bioweapons. To address these biosecurity challenges, we introduce two innovative watermarking techniques to reliably track the designed DNA: DNAMark and CentralMark. DNAMark employs synonymous codon substitutions to embed watermarks in DNA sequences while preserving the original function. CentralMark further advances this by creating inheritable watermarks that transfer from DNA to translated proteins, leveraging protein embeddings to ensure detection across the central dogma. Both methods utilize semantic embeddings to generate watermark logits, enhancing robustness against natural mutations, synthesis errors, and adversarial attacks. Evaluated on our therapeutic DNA benchmark, DNAMark and CentralMark achieve F1 detection scores above 0.85 under various conditions, while maintaining over 60% sequence similarity to ground truth and degeneracy scores below 15%. A case study on the CRISPR-Cas9 system underscores CentralMark's utility in real-world settings. This work establishes a vital framework for securing DNA language models, balancing innovation with accountability to mitigate biosecurity risks.

摘要: DNA语言模型以前所未有的精确度彻底改变了我们理解和设计DNA序列（生命的基本语言）的能力，使治疗学、合成生物学和基因编辑领域的变革性应用成为可能。然而，这种能力也带来了巨大的双重用途风险，包括产生病原体、病毒甚至生物武器的可能性。为了应对这些生物安全挑战，我们引入了两种创新的水印技术来可靠地跟踪设计的DNA：DNAMark和CentralMark。DNAMark采用同义密码子替换在DNA序列中嵌入水印，同时保留原始功能。CentralMark通过创建从DNA转移到翻译蛋白质的可遗传水印进一步推进了这一点，利用蛋白质嵌入来确保跨中心教条的检测。这两种方法都利用语义嵌入来生成水印日志，增强了针对自然突变、合成错误和对抗攻击的鲁棒性。根据我们的治疗性DNA基准进行评估，DNAMark和CentralMark在各种条件下的F1检测评分均高于0.85，同时与基本真相保持60%以上的序列相似性，简并度评分低于15%。关于CRISPR-Cas9系统的案例研究强调了CentralMark在现实世界环境中的实用性。这项工作为保护DNA语言模型建立了一个重要的框架，平衡创新与问责制以减轻生物安全风险。



## **34. On the Robustness of RSMA to Adversarial BD-RIS-Induced Interference**

RMA对对抗性BD-RIS诱导干扰的鲁棒性 eess.SP

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2505.20146v2) [paper-pdf](http://arxiv.org/pdf/2505.20146v2)

**Authors**: Arthur S. de Sena, Jacek Kibilda, Nurul H. Mahmood, Andre Gomes, Luiz A. DaSilva, Matti Latva-aho

**Abstract**: This article investigates the robustness of rate-splitting multiple access (RSMA) in multi-user multiple-input single-output (MISO) systems to interference attacks against channel acquisition induced by beyond-diagonal RISs (BD-RISs). Two primary attack strategies, random and aligned interference, are proposed for fully connected and group-connected reconfigurable intelligent surface (RIS) architectures. Valid random reflection coefficients are generated exploiting the Takagi factorization, while potent aligned interference attacks are achieved through optimization strategies based on a quadratically constrained quadratic program (QCQP) reformulation followed by projections onto the unitary manifold. Our numerical findings reveal that, when perfect channel state information (CSI) is available, RSMA behaves similarly to space-division multiple access (SDMA) and thus is highly susceptible to the attack, with BD-RIS inducing severe performance loss and significantly outperforming diagonal RIS. However, under imperfect CSI, RSMA consistently demonstrates significantly greater robustness than SDMA, particularly as the system's transmit power increases.

摘要: 本文研究了多用户多输入单输出（MISO）系统中的速率分裂多址（RSM）对超对角RIS（BD-RIS）引起的针对通道捕获的干扰攻击的鲁棒性。针对全连接和群连接可重构智能表面（RIS）架构，提出了两种主要的攻击策略：随机干扰和对齐干扰。有效的随机反射系数是利用Takagi因式分解生成的，而有效的对齐干扰攻击是通过基于二次约束二次规划（QCQP）重新公式的优化策略来实现的，然后是投影到正总管上。我们的数字研究结果表明，当完美通道状态信息（SI）可用时，RMA的行为与空间分多址（SBA）类似，因此极易受到攻击，其中BD-RIS会导致严重的性能损失，并且性能显着优于对角线RIS。然而，在不完美的SI下，RMA始终表现出比SBA明显更高的鲁棒性，特别是当系统发射功率增加时。



## **35. ADVEDM:Fine-grained Adversarial Attack against VLM-based Embodied Agents**

ADVEDM：针对基于VLM的排队代理的细粒度对抗攻击 cs.CV

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.16645v1) [paper-pdf](http://arxiv.org/pdf/2509.16645v1)

**Authors**: Yichen Wang, Hangtao Zhang, Hewen Pan, Ziqi Zhou, Xianlong Wang, Peijin Guo, Lulu Xue, Shengshan Hu, Minghui Li, Leo Yu Zhang

**Abstract**: Vision-Language Models (VLMs), with their strong reasoning and planning capabilities, are widely used in embodied decision-making (EDM) tasks in embodied agents, such as autonomous driving and robotic manipulation. Recent research has increasingly explored adversarial attacks on VLMs to reveal their vulnerabilities. However, these attacks either rely on overly strong assumptions, requiring full knowledge of the victim VLM, which is impractical for attacking VLM-based agents, or exhibit limited effectiveness. The latter stems from disrupting most semantic information in the image, which leads to a misalignment between the perception and the task context defined by system prompts. This inconsistency interrupts the VLM's reasoning process, resulting in invalid outputs that fail to affect interactions in the physical world. To this end, we propose a fine-grained adversarial attack framework, ADVEDM, which modifies the VLM's perception of only a few key objects while preserving the semantics of the remaining regions. This attack effectively reduces conflicts with the task context, making VLMs output valid but incorrect decisions and affecting the actions of agents, thus posing a more substantial safety threat in the physical world. We design two variants of based on this framework, ADVEDM-R and ADVEDM-A, which respectively remove the semantics of a specific object from the image and add the semantics of a new object into the image. The experimental results in both general scenarios and EDM tasks demonstrate fine-grained control and excellent attack performance.

摘要: 视觉语言模型（VLMS）以其强大的推理和规划能力，广泛应用于嵌入式智能体中的嵌入式决策（EDA）任务，例如自动驾驶和机器人操纵。最近的研究越来越多地探索对VLM的对抗攻击，以揭示它们的漏洞。然而，这些攻击要么依赖于过于强的假设，需要完全了解受害者VLM，这对于攻击基于LM的代理来说是不切实际的，要么表现出有限的有效性。后者源于扰乱图像中的大多数语义信息，从而导致感知与系统提示定义的任务上下文之间的不一致。这种不一致性会中断VLM的推理过程，导致无效输出，无法影响物理世界中的交互。为此，我们提出了一个细粒度的对抗攻击框架ADVEDM，它仅修改VLM对少数关键对象的感知，同时保留其余区域的语义。这种攻击有效地减少了与任务上下文的冲突，使VLM输出有效但不正确的决策并影响代理的行为，从而对物理世界构成更大的安全威胁。我们基于该框架设计了两个变体ADVEDM-R和ADVEDM-A，它们分别从图像中删除特定对象的语义并将新对象的语义添加到图像中。一般场景和EDA任务中的实验结果都证明了细粒度控制和出色的攻击性能。



## **36. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

糖衣毒药：良性一代解锁法学硕士越狱 cs.CR

Accepted by EMNLP2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2504.05652v3) [paper-pdf](http://arxiv.org/pdf/2504.05652v3)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Hao Zhang, Jia-Chen Zhang, Zheng Zhou

**Abstract**: With the increasingly deep integration of large language models (LLMs) across diverse domains, the effectiveness of their safety mechanisms is encountering severe challenges. Currently, jailbreak attacks based on prompt engineering have become a major safety threat. However, existing methods primarily rely on black-box manipulation of prompt templates, resulting in poor interpretability and limited generalization. To break through the bottleneck, this study first introduces the concept of Defense Threshold Decay (DTD), revealing the potential safety impact caused by LLMs' benign generation: as benign content generation in LLMs increases, the model's focus on input instructions progressively diminishes. Building on this insight, we propose the Sugar-Coated Poison (SCP) attack paradigm, which uses a "semantic reversal" strategy to craft benign inputs that are opposite in meaning to malicious intent. This strategy induces the models to generate extensive benign content, thereby enabling adversarial reasoning to bypass safety mechanisms. Experiments show that SCP outperforms existing baselines. Remarkably, it achieves an average attack success rate of 87.23% across six LLMs. For defense, we propose Part-of-Speech Defense (POSD), leveraging verb-noun dependencies for syntactic analysis to enhance safety of LLMs while preserving their generalization ability.

摘要: 随着大型语言模型（LLM）跨不同领域的日益深入集成，其安全机制的有效性面临严峻挑战。目前，基于即时工程的越狱攻击已成为重大安全威胁。然而，现有的方法主要依赖于提示模板的黑匣子操作，导致可解释性较差且概括性有限。为了突破瓶颈，本研究首先引入了防御阈值衰变（DART）的概念，揭示了LLM良性生成对安全的潜在影响：随着LLM良性内容生成的增加，模型对输入指令的关注逐渐减少。基于这一见解，我们提出了糖衣毒药（SCP）攻击范式，该范式使用“语义逆转”策略来制造与恶意意图含义相反的良性输入。该策略促使模型生成广泛的良性内容，从而使对抗推理能够绕过安全机制。实验表明SCP优于现有基线。值得注意的是，它在六个LLM中的平均攻击成功率为87.23%。对于防御，我们提出了词性防御（POSD），利用动词-名词依赖进行语法分析，以增强LLM的安全性，同时保留其概括能力。



## **37. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

个人可以操纵多主体的集体决策吗？ cs.CL

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.16494v1) [paper-pdf](http://arxiv.org/pdf/2509.16494v1)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.

摘要: 个体大型语言模型（LLM）已在医疗保健和法律等各个领域展现出强大的能力。最近的研究还表明，协调的多智能体系统通过协作表现出增强的决策和推理能力。然而，由于单个LLM的脆弱性以及访问多代理系统中所有代理的困难，出现了一个关键问题：如果攻击者只知道一个代理，他们还能生成能够误导集体决策的对抗样本吗？为了探索这个问题，我们将其描述为一个信息不完整的游戏，其中攻击者只知道一个目标代理，并且缺乏对系统中其他代理的了解。通过这个公式，我们提出了M-Spoiler，这是一个模拟多智能体系统内的智能体交互以生成对抗样本的框架。然后使用这些样本来操纵目标系统中的目标代理，误导系统的协作决策过程。更具体地说，M-Spoiler引入了一种顽固代理，它通过模拟目标系统中代理的潜在顽固反应来积极帮助优化对抗样本。这增强了生成的对抗样本误导系统的有效性。通过针对各种任务的广泛实验，我们的研究结果证实了多代理系统中单个代理的知识所带来的风险，并证明了我们框架的有效性。我们还探索了几种防御机制，表明我们提出的攻击框架仍然比基线更有效，强调了进一步研究防御策略的必要性。



## **38. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

目标对齐：提取对齐的LLM的安全分类器 cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2501.16534v2) [paper-pdf](http://arxiv.org/pdf/2501.16534v2)

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.

摘要: 大型语言模型（LLM）中的对齐用于强制执行安全等准则。然而，面对越狱攻击，调整失败了，这些攻击修改了输入以引发不安全的输出。本文介绍并评估了一种新的越狱攻击技术。我们观察到，对齐在LLM中嵌入了一个安全分类器，负责在拒绝和合规之间做出决定，并试图提取该分类器的近似值：代理分类器。为此，我们从LLM的子集中构建候选分类器。我们首先评估候选分类器在良性和对抗环境中接近LLM安全分类器的程度。然后，我们攻击候选人并衡量由此产生的对抗输入转移到LLM的程度。我们的评估表明，最好的候选人只需使用20%的模型架构即可实现准确的一致性（F1评分高于80%）。此外，我们发现，安装在代理分类器上的攻击可以转移到LLM，具有很高的成功率。例如，仅使用50%的Llama 2模型的代理实现了70%的攻击成功率（ASR），而内存占用和运行时间只有一半-与直接攻击LLM相比有了实质性的改进，我们只观察到22%的ASR。这些结果表明，提取代理分类器是一种有效和高效的手段建模（并在其中解决）的漏洞对齐模型越狱攻击。



## **39. Secure Confidential Business Information When Sharing Machine Learning Models**

共享机器学习模型时保护机密业务信息 cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16352v1) [paper-pdf](http://arxiv.org/pdf/2509.16352v1)

**Authors**: Yunfan Yang, Jiarong Xu, Hongzhe Zhang, Xiao Fang

**Abstract**: Model-sharing offers significant business value by enabling firms with well-established Machine Learning (ML) models to monetize and share their models with others who lack the resources to develop ML models from scratch. However, concerns over data confidentiality remain a significant barrier to model-sharing adoption, as Confidential Property Inference (CPI) attacks can exploit shared ML models to uncover confidential properties of the model provider's private model training data. Existing defenses often assume that CPI attacks are non-adaptive to the specific ML model they are targeting. This assumption overlooks a key characteristic of real-world adversaries: their responsiveness, i.e., adversaries' ability to dynamically adjust their attack models based on the information of the target and its defenses. To overcome this limitation, we propose a novel defense method that explicitly accounts for the responsive nature of real-world adversaries via two methodological innovations: a novel Responsive CPI attack and an attack-defense arms race framework. The former emulates the responsive behaviors of adversaries in the real world, and the latter iteratively enhances both the target and attack models, ultimately producing a secure ML model that is robust against responsive CPI attacks. Furthermore, we propose and integrate a novel approximate strategy into our defense, which addresses a critical computational bottleneck of defense methods and improves defense efficiency. Through extensive empirical evaluations across various realistic model-sharing scenarios, we demonstrate that our method outperforms existing defenses by more effectively defending against CPI attacks, preserving ML model utility, and reducing computational overhead.

摘要: 模型共享使拥有成熟机器学习（ML）模型的公司能够将其模型货币化并与缺乏从头开始开发ML模型资源的其他公司共享，从而提供了显着的商业价值。然而，对数据机密性的担忧仍然是模型共享采用的一个重大障碍，因为机密财产推理（CPI）攻击可以利用共享ML模型来揭露模型提供商私人模型训练数据的机密属性。现有的防御系统通常假设CPI攻击不适应其目标的特定ML模型。这一假设忽视了现实世界对手的一个关键特征：他们的响应能力，即对手根据目标及其防御信息动态调整攻击模型的能力。为了克服这一限制，我们提出了一种新型防御方法，通过两种方法创新明确解释了现实世界对手的响应性质：新型响应CPI攻击和攻击防御军备竞赛框架。前者模拟现实世界中对手的响应行为，后者迭代增强目标和攻击模型，最终生成一个针对响应CPI攻击稳健的安全ML模型。此外，我们提出了一种新型的近似策略并将其集成到我们的防御中，该策略解决了防御方法的关键计算瓶颈并提高了防御效率。通过对各种现实模型共享场景进行广泛的经验评估，我们证明我们的方法通过更有效地防御CPI攻击、保留ML模型效用和减少计算负担来优于现有的防御。



## **40. Robust Vision-Language Models via Tensor Decomposition: A Defense Against Adversarial Attacks**

通过张量分解构建稳健的视觉语言模型：对抗性攻击的防御 cs.CV

To be presented as a poster at the Workshop on Safe and Trustworthy  Multimodal AI Systems (SafeMM-AI), 2025

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16163v1) [paper-pdf](http://arxiv.org/pdf/2509.16163v1)

**Authors**: Het Patel, Muzammil Allie, Qian Zhang, Jia Chen, Evangelos E. Papalexakis

**Abstract**: Vision language models (VLMs) excel in multimodal understanding but are prone to adversarial attacks. Existing defenses often demand costly retraining or significant architecture changes. We introduce a lightweight defense using tensor decomposition suitable for any pre-trained VLM, requiring no retraining. By decomposing and reconstructing vision encoder representations, it filters adversarial noise while preserving meaning. Experiments with CLIP on COCO and Flickr30K show improved robustness. On Flickr30K, it restores 12.3\% performance lost to attacks, raising Recall@1 accuracy from 7.5\% to 19.8\%. On COCO, it recovers 8.1\% performance, improving accuracy from 3.8\% to 11.9\%. Analysis shows Tensor Train decomposition with low rank (8-32) and low residual strength ($\alpha=0.1-0.2$) is optimal. This method is a practical, plug-and-play solution with minimal overhead for existing VLMs.

摘要: 视觉语言模型（VLM）在多模式理解方面表现出色，但容易受到对抗性攻击。现有的防御通常需要昂贵的再培训或重大的架构更改。我们引入了一种使用张量分解的轻量级防御，适合任何预训练的VLM，无需重新训练。通过分解和重建视觉编码器表示，它过滤对抗性噪音，同时保留意义。COCO和Flickr 30 K上的CLIP实验显示出鲁棒性得到了改善。在Flickr 30 K上，它恢复了因攻击而损失的12.3%性能，将Recall@1准确率从7.5%提高到19.8%。在COCO上，它恢复了8.1%的性能，准确性从3.8%提高到11.9%。分析表明，低等级（8-32）和低剩余强度（$\Alpha=0.1-0.2$）的张量列车分解是最佳的。这种方法是一种实用的、即插即用的解决方案，对于现有的VLM来说具有最小的负担。



## **41. Randomized Smoothing Meets Vision-Language Models**

随机平滑满足视觉语言模型 cs.LG

EMNLP'25 full version, including appendix (proofs, additional  experiments)

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16088v1) [paper-pdf](http://arxiv.org/pdf/2509.16088v1)

**Authors**: Emmanouil Seferis, Changshun Wu, Stefanos Kollias, Saddek Bensalem, Chih-Hong Cheng

**Abstract**: Randomized smoothing (RS) is one of the prominent techniques to ensure the correctness of machine learning models, where point-wise robustness certificates can be derived analytically. While RS is well understood for classification, its application to generative models is unclear, since their outputs are sequences rather than labels. We resolve this by connecting generative outputs to an oracle classification task and showing that RS can still be enabled: the final response can be classified as a discrete action (e.g., service-robot commands in VLAs), as harmful vs. harmless (content moderation or toxicity detection in VLMs), or even applying oracles to cluster answers into semantically equivalent ones. Provided that the error rate for the oracle classifier comparison is bounded, we develop the theory that associates the number of samples with the corresponding robustness radius. We further derive improved scaling laws analytically relating the certified radius and accuracy to the number of samples, showing that the earlier result of 2 to 3 orders of magnitude fewer samples sufficing with minimal loss remains valid even under weaker assumptions. Together, these advances make robustness certification both well-defined and computationally feasible for state-of-the-art VLMs, as validated against recent jailbreak-style adversarial attacks.

摘要: 随机平滑（RS）是确保机器学习模型正确性的重要技术之一，其中可以通过分析推导逐点鲁棒性证书。虽然RS对于分类的理解很好，但它在生成模型中的应用尚不清楚，因为它们的输出是序列而不是标签。我们通过将生成性输出连接到Oracle分类任务并表明RS仍然可以启用来解决这个问题：最终响应可以被分类为离散动作（例如，VLA中的服务机器人命令），有害与无害（VLM中的内容审核或毒性检测），甚至应用Oracle将答案聚集成语义等效的答案。假设Oracle分类器比较的错误率是有界的，我们开发了将样本数量与相应的鲁棒性半径联系起来的理论。我们进一步推导出改进的缩放定律，将认证半径和准确性与样本数量联系起来，表明即使在较弱的假设下，早期的2到3个数量级的样本数量就足以达到最小损失的结果仍然有效。总而言之，这些进步使得最先进的VLM的稳健性认证定义明确且在计算上可行，并针对最近的越狱式对抗攻击进行了验证。



## **42. Attention Schema-based Attention Control (ASAC): A Cognitive-Inspired Approach for Attention Management in Transformers**

基于注意力方案的注意力控制（ASAC）：变形金刚中一种受认知启发的注意力管理方法 cs.AI

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16058v1) [paper-pdf](http://arxiv.org/pdf/2509.16058v1)

**Authors**: Krati Saxena, Federico Jurado Ruiz, Guido Manzi, Dianbo Liu, Alex Lamb

**Abstract**: Attention mechanisms have become integral in AI, significantly enhancing model performance and scalability by drawing inspiration from human cognition. Concurrently, the Attention Schema Theory (AST) in cognitive science posits that individuals manage their attention by creating a model of the attention itself, effectively allocating cognitive resources. Inspired by AST, we introduce ASAC (Attention Schema-based Attention Control), which integrates the attention schema concept into artificial neural networks. Our initial experiments focused on embedding the ASAC module within transformer architectures. This module employs a Vector-Quantized Variational AutoEncoder (VQVAE) as both an attention abstractor and controller, facilitating precise attention management. By explicitly modeling attention allocation, our approach aims to enhance system efficiency. We demonstrate ASAC's effectiveness in both the vision and NLP domains, highlighting its ability to improve classification accuracy and expedite the learning process. Our experiments with vision transformers across various datasets illustrate that the attention controller not only boosts classification accuracy but also accelerates learning. Furthermore, we have demonstrated the model's robustness and generalization capabilities across noisy and out-of-distribution datasets. In addition, we have showcased improved performance in multi-task settings. Quick experiments reveal that the attention schema-based module enhances resilience to adversarial attacks, optimizes attention to improve learning efficiency, and facilitates effective transfer learning and learning from fewer examples. These promising results establish a connection between cognitive science and machine learning, shedding light on the efficient utilization of attention mechanisms in AI systems.

摘要: 注意力机制已成为人工智能中不可或缺的一部分，通过从人类认知中汲取灵感，显着增强模型性能和可扩展性。与此同时，认知科学中的注意力图式理论（AST）认为，个人通过创建注意力本身的模型来管理注意力，有效地分配认知资源。受AST的启发，我们引入了ASAC（基于注意力模式的注意力控制），它将注意力模式概念集成到人工神经网络中。我们最初的实验重点是将ASAC模块嵌入到Transformer架构中。该模块采用Vector-Quantized Variational AutoEncoder（VQVAE）作为注意力抽象器和控制器，促进精确的注意力管理。通过显式建模注意力分配，我们的方法旨在提高系统效率。我们展示了ASAC在视觉和NLP领域的有效性，强调了其提高分类准确性和加快学习过程的能力。我们在各种数据集中对视觉转换器进行的实验表明，注意力控制器不仅提高了分类准确性，而且还加速了学习。此外，我们还展示了该模型在有噪和非分布数据集中的稳健性和概括能力。此外，我们还展示了在多任务设置中改进的性能。快速实验表明，基于注意力方案的模块增强了对对抗攻击的弹性，优化注意力以提高学习效率，并促进有效的迁移学习和从更少的示例中学习。这些有希望的结果建立了认知科学和机器学习之间的联系，揭示了人工智能系统中注意力机制的有效利用。



## **43. Swarm Oracle: Trustless Blockchain Agreements through Robot Swarms**

Swarm Oracle：通过机器人群达成不可信的区块链协议 cs.RO

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15956v1) [paper-pdf](http://arxiv.org/pdf/2509.15956v1)

**Authors**: Alexandre Pacheco, Hanqing Zhao, Volker Strobel, Tarik Roukny, Gregory Dudek, Andreagiovanni Reina, Marco Dorigo

**Abstract**: Blockchain consensus, rooted in the principle ``don't trust, verify'', limits access to real-world data, which may be ambiguous or inaccessible to some participants. Oracles address this limitation by supplying data to blockchains, but existing solutions may reduce autonomy, transparency, or reintroduce the need for trust. We propose Swarm Oracle: a decentralized network of autonomous robots -- that is, a robot swarm -- that use onboard sensors and peer-to-peer communication to collectively verify real-world data and provide it to smart contracts on public blockchains. Swarm Oracle leverages the built-in decentralization, fault tolerance and mobility of robot swarms, which can flexibly adapt to meet information requests on-demand, even in remote locations. Unlike typical cooperative robot swarms, Swarm Oracle integrates robots from multiple stakeholders, protecting the system from single-party biases but also introducing potential adversarial behavior. To ensure the secure, trustless and global consensus required by blockchains, we employ a Byzantine fault-tolerant protocol that enables robots from different stakeholders to operate together, reaching social agreements of higher quality than the estimates of individual robots. Through extensive experiments using both real and simulated robots, we showcase how consensus on uncertain environmental information can be achieved, despite several types of attacks orchestrated by large proportions of the robots, and how a reputation system based on blockchain tokens lets Swarm Oracle autonomously recover from faults and attacks, a requirement for long-term operation.

摘要: 区块链共识植根于“不信任、不验证”原则，限制了对现实世界数据的访问，而这些数据可能不明确或对一些参与者来说无法访问。Oracle通过向区块链提供数据来解决这一限制，但现有的解决方案可能会降低自主性、透明度，或重新引入信任的需求。我们建议Swarm Oracle：自主机器人的去中心化网络（即机器人群）使用机载传感器和点对点通信来集体验证现实世界的数据并将其提供给公共区块链上的智能合同。Swarm Oracle利用机器人群的内置去中心化、故障容忍和移动性，可以灵活调整以满足按需的信息请求，即使是在远程位置。与典型的合作机器人群不同，Swarm Oracle集成了来自多个利益相关者的机器人，保护系统免受一方偏见的影响，但也引入了潜在的对抗行为。为了确保区块链所需的安全、不信任和全球共识，我们采用了一种拜占庭式的故障容忍协议，使来自不同利益相关者的机器人能够共同操作，达成比单个机器人估计质量更高的社会协议。通过使用真实和模拟机器人的广泛实验，我们展示了尽管大部分机器人策划了多种类型的攻击，但如何就不确定的环境信息达成共识，以及基于区块链代币的声誉系统如何让Swarm Oracle自主从故障和攻击中恢复，这是长期运营的要求。



## **44. Bridging Batch and Streaming Estimations to System Identification under Adversarial Attacks**

对抗性攻击下将批处理和流估计连接到系统识别 math.OC

15 pages, 2 figures

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15794v1) [paper-pdf](http://arxiv.org/pdf/2509.15794v1)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: System identification in modern engineering systems faces emerging challenges from unanticipated adversarial attacks beyond existing detection mechanisms. In this work, we obtain a provably accurate estimate of the Markov parameter matrix of order $k$ to identify partially observed linear systems, in which the probability of having an attack at each time is $O(1/k)$. We show that given the batch data accumulated up to time $T^*$, the $\ell_2$-norm estimator achieves an error decaying exponentially as $k$ grows. We then propose a stochastic projected subgradient descent algorithm on streaming data that produces an estimate at each time $t<T^*$, in which case the expected estimation error proves to be the larger of $O(k/\sqrt{t})$ and an exponentially decaying term in $k$. This stochastic approach illustrates how non-smooth estimators can leverage first-order methods despite lacking recursive formulas. Finally, we integrate batch and streaming estimations to recover the Hankel matrix using the appropriate estimates of the Markov parameter matrix, which enables the synthesis of a robust adaptive controller based on the estimated balanced truncated model under adversarial attacks.

摘要: 现代工程系统中的系统识别面临着来自现有检测机制之外的意想不到的对抗攻击的新挑战。在这项工作中，我们获得了$k$阶马尔科夫参数矩阵的可证明准确的估计，以识别部分观察线性系统，其中每次遭受攻击的概率为$O（1/k）$。我们表明，给定到$T^*$累积的批数据，$\ell_2$-norm估计器会实现随着$k$增长而指数级衰减的误差。然后，我们在流数据上提出了一种随机投影次梯度下降算法，该算法在每次$t<T &*$时产生一个估计，在这种情况下，预期估计误差被证明是$O（k/\SQRT{t}）$和$k$中的指数衰减项中的较大者。这种随机方法说明了非光滑估计器如何在缺乏迭代公式的情况下利用一阶方法。最后，我们集成批量估计和流估计，以使用Markov参数矩阵的适当估计来恢复汉克尔矩阵，这使得能够在对抗攻击下基于估计的平衡截断模型来合成鲁棒自适应控制器。



## **45. Explainable Deep Learning Based Adversarial Defense for Automatic Modulation Classification**

基于可解释深度学习的自动调制分类对抗防御 eess.SP

Accepted by IEEE Internet of Things Journal

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15766v1) [paper-pdf](http://arxiv.org/pdf/2509.15766v1)

**Authors**: Peihao Dong, Jingchun Wang, Shen Gao, Fuhui Zhou, Qihui Wu

**Abstract**: Deep learning (DL) has been widely applied to enhance automatic modulation classification (AMC). However, the elaborate AMC neural networks are susceptible to various adversarial attacks, which are challenging to handle due to the generalization capability and computational cost. In this article, an explainable DL based defense scheme, called SHapley Additive exPlanation enhanced Adversarial Fine-Tuning (SHAP-AFT), is developed in the perspective of disclosing the attacking impact on the AMC network. By introducing the concept of cognitive negative information, the motivation of using SHAP for defense is theoretically analyzed first. The proposed scheme includes three stages, i.e., the attack detection, the information importance evaluation, and the AFT. The first stage indicates the existence of the attack. The second stage evaluates contributions of the received data and removes those data positions using negative Shapley values corresponding to the dominating negative information caused by the attack. Then the AMC network is fine-tuned based on adversarial adaptation samples using the refined received data pattern. Simulation results show the effectiveness of the Shapley value as the key indicator as well as the superior defense performance of the proposed SHAP-AFT scheme in face of different attack types and intensities.

摘要: 深度学习（DL）已广泛应用于增强自动调制分类（AMC）。然而，精心设计的AMC神经网络容易受到各种对抗攻击，由于概括能力和计算成本，这些攻击的处理具有挑战性。本文从揭示攻击对AMC网络的影响的角度出发，开发了一种可解释的基于DL的防御方案，称为SHAP-AFT，称为SHAP-AFT。通过引入认知负信息的概念，首先从理论上分析了使用SHAP进行防御的动机。拟议的方案包括三个阶段，即攻击检测、信息重要性评估和AFT。第一阶段表明攻击的存在。第二阶段评估接收到的数据的贡献，并使用与攻击引起的主要负面信息相对应的负Shapley值来删除这些数据位置。然后，使用改进的接收数据模式，根据对抗适应样本对AMC网络进行微调。仿真结果表明了Shapley值作为关键指标的有效性，以及所提出的SHAP-AFT方案在面对不同类型和强度的攻击时具有优越的防御性能。



## **46. An Adversarial Robust Behavior Sequence Anomaly Detection Approach Based on Critical Behavior Unit Learning**

基于关键行为单元学习的对抗性鲁棒行为序列异常检测方法 cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15756v1) [paper-pdf](http://arxiv.org/pdf/2509.15756v1)

**Authors**: Dongyang Zhan, Kai Tan, Lin Ye, Xiangzhan Yu, Hongli Zhang, Zheng He

**Abstract**: Sequential deep learning models (e.g., RNN and LSTM) can learn the sequence features of software behaviors, such as API or syscall sequences. However, recent studies have shown that these deep learning-based approaches are vulnerable to adversarial samples. Attackers can use adversarial samples to change the sequential characteristics of behavior sequences and mislead malware classifiers. In this paper, an adversarial robustness anomaly detection method based on the analysis of behavior units is proposed to overcome this problem. We extract related behaviors that usually perform a behavior intention as a behavior unit, which contains the representative semantic information of local behaviors and can be used to improve the robustness of behavior analysis. By learning the overall semantics of each behavior unit and the contextual relationships among behavior units based on a multilevel deep learning model, our approach can mitigate perturbation attacks that target local and large-scale behaviors. In addition, our approach can be applied to both low-level and high-level behavior logs (e.g., API and syscall logs). The experimental results show that our approach outperforms all the compared methods, which indicates that our approach has better performance against obfuscation attacks.

摘要: 顺序深度学习模型（例如，RNN和LSTM）可以学习软件行为的序列特征，例如API或syscall序列。然而，最近的研究表明，这些基于深度学习的方法很容易受到对抗性样本的影响。攻击者可以使用对抗样本来改变行为序列的顺序特征并误导恶意软件分类器。针对这一问题，本文提出了一种基于行为单元分析的对抗鲁棒性异常检测方法。我们提取通常执行行为意图的相关行为作为行为单元，其中包含局部行为的代表性语义信息，可用于提高行为分析的鲁棒性。通过基于多层深度学习模型学习每个行为单元的整体语义以及行为单元之间的上下文关系，我们的方法可以减轻针对局部和大规模行为的扰动攻击。此外，我们的方法可以应用于低级和高级行为日志（例如，API和系统调用日志）。实验结果表明，我们的方法优于所有比较方法，这表明我们的方法在对抗模糊攻击方面具有更好的性能。



## **47. Chernoff Information as a Privacy Constraint for Adversarial Classification and Membership Advantage**

冲突信息作为对抗性分类和成员优势的隐私约束 cs.IT

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2403.10307v4) [paper-pdf](http://arxiv.org/pdf/2403.10307v4)

**Authors**: Ayşe Ünsal

**Abstract**: This work inspects a privacy metric based on Chernoff information, namely Chernoff differential privacy, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we characterize the relationship between $\varepsilon\textrm{-}$differential privacy, the best error exponent of one of the errors (when the other is fixed) and the best average error exponent. Accordingly, we re-derive Chernoff differential privacy in connection with $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative, and prove its relation with Kullback-Leibler (KL) differential privacy. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$ and the impact of the adversary's attack in Laplace mechanisms. Lastly, we introduce a new upper bound on adversary's membership advantage in membership inference attacks using Chernoff DP and numerically compare its performance with existing alternatives based on $(\varepsilon, \delta)\textrm{-}$differential privacy in the literature.

摘要: 这项工作检查了基于Timoff信息的隐私指标，即Timoff差异隐私，因为它在描述最佳分类器性能方面具有重要意义。对抗性分类，就像任何其他分类问题一样，在二元分类的情况下，都是围绕决定任何一个类别时（平均或正确检测）错误概率的最小化而建立的。与经典假设测试问题不同，其中虚警和误判概率分别处理，导致最佳错误指数的不对称行为，在这项工作中，我们描述了$\varepð\texttrm {-}$差异隐私、其中一个错误的最佳错误指数（当另一个错误是固定的时）和最佳平均错误指数之间的关系。因此，我们使用Radon-Nikodym衍生物重新推导与$\varepð\textrm{-}$差异隐私相关的Timoff差异隐私，并证明其与Kullback-Leibler（KL）差异隐私的关系。随后，我们给出了数值评估结果，该结果表明，作为隐私参数$\varepð $和对手攻击在拉普拉斯机制中的影响的函数，Lattoff信息优于Kullback-Leibler分歧。最后，我们在使用Deliverff DP的成员资格推断攻击中引入了对手成员资格优势的新上限，并将其性能与文献中基于$（\varepð，\delta）\texttrm {-}$差异隐私的现有替代方案进行了数字比较。



## **48. Inference Attacks on Encrypted Online Voting via Traffic Analysis**

通过流量分析对加密在线投票的推理攻击 cs.CR

Accepted at ISC 2025

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15694v1) [paper-pdf](http://arxiv.org/pdf/2509.15694v1)

**Authors**: Anastasiia Belousova, Francesco Marchiori, Mauro Conti

**Abstract**: Online voting enables individuals to participate in elections remotely, offering greater efficiency and accessibility in both governmental and organizational settings. As this method gains popularity, ensuring the security of online voting systems becomes increasingly vital, as the systems supporting it must satisfy a demanding set of security requirements. Most research in this area emphasizes the design and verification of cryptographic protocols to protect voter integrity and system confidentiality. However, other vectors, such as network traffic analysis, remain relatively understudied, even though they may pose significant threats to voter privacy and the overall trustworthiness of the system.   In this paper, we examine how adversaries can exploit metadata from encrypted network traffic to uncover sensitive information during online voting. Our analysis reveals that, even without accessing the encrypted content, it is possible to infer critical voter actions, such as whether a person votes, the exact moment a ballot is submitted, and whether the ballot is valid or spoiled. We test these attacks with both rule-based techniques and machine learning methods. We evaluate our attacks on two widely used online voting platforms, one proprietary and one partially open source, achieving classification accuracy as high as 99.5%. These results expose a significant privacy vulnerability that threatens key properties of secure elections, including voter secrecy and protection against coercion or vote-buying. We explore mitigations to our attacks, demonstrating that countermeasures such as payload padding and timestamp equalization can substantially limit their effectiveness.

摘要: 在线投票使个人能够远程参与选举，从而在政府和组织环境中提供更高的效率和可访问性。随着这种方法的普及，确保在线投票系统的安全性变得越来越重要，因为支持它的系统必须满足一系列严格的安全要求。该领域的大多数研究都强调密码协议的设计和验证，以保护选民完整性和系统机密性。然而，网络流量分析等其他载体的研究仍然相对不足，尽管它们可能对选民隐私和系统的整体可信度构成重大威胁。   在本文中，我们研究了对手如何利用加密网络流量中的元数据来发现在线投票期间的敏感信息。我们的分析表明，即使不访问加密内容，也可以推断出关键的选民行为，例如一个人是否投票、提交选票的确切时刻以及选票是否有效或损坏。我们使用基于规则的技术和机器学习方法来测试这些攻击。我们评估了对两个广泛使用的在线投票平台的攻击，一个是专有的，一个是部分开源的，实现了高达99.5%的分类准确率。这些结果暴露了一个严重的隐私漏洞，该漏洞威胁到安全选举的关键属性，包括选民保密性和防止胁迫或收买选票。我们探索了攻击的缓解措施，证明有效负载填充和时间戳均衡等对策可以极大地限制其有效性。



## **49. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

AdaSteer：您的LLM本质上是一个适应性越狱捍卫者 cs.CR

19 pages, 6 figures, 10 tables

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2504.09466v2) [paper-pdf](http://arxiv.org/pdf/2504.09466v2)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.

摘要: 尽管在安全调整方面做出了广泛的努力，但大型语言模型（LLM）仍然容易受到越狱攻击。激活转向提供了一种无需训练的防御方法，但依赖于固定的转向系数，从而导致次优保护和良性输入的错误拒绝增加。为了解决这个问题，我们提出了AdaSteer，这是一种自适应激活引导方法，可以根据输入特征动态调整模型行为。我们确定了两个关键属性：拒绝定律（R-Law），它表明与拒绝方向相反的越狱输入需要更强的引导，以及有害定律（H-Law），它区分对抗性和良性输入。AdaSteer沿着拒绝方向（RD）和有害方向（HD）引导输入表示，通过逻辑回归学习自适应系数，确保强大的越狱防御，同时保持良性的输入处理。在LLaMA-3.1、Gemma-2和Qwen2.5上的实验表明，AdaSteer在多种越狱攻击中的性能优于基线方法，对效用的影响最小。我们的研究结果突出了可解释的模型内部实时，灵活的安全执法LLM的潜力。



## **50. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm**

DNA-DetectLLM：通过DNA启发的突变修复范式揭示人工智能生成的文本 cs.CL

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15550v1) [paper-pdf](http://arxiv.org/pdf/2509.15550v1)

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao

**Abstract**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets.

摘要: 大型语言模型（LLM）的快速发展模糊了人工智能生成的文本和人类编写的文本之间的界限。这一进展带来了错误信息、作者身份模糊和知识产权问题等社会风险，凸显了对可靠的人工智能生成文本检测方法的迫切需求。然而，生成式语言建模的最新进展导致人类书写文本和人工智能生成文本的特征分布之间存在显着重叠，模糊了分类边界，并使准确检测变得越来越具有挑战性。为了解决上述挑战，我们提出了一种受DNA启发的视角，利用基于修复的流程来直接且可解释地捕捉人类书写和人工智能生成的文本之间的内在差异。基于这一观点，我们引入了DNA-DetectLLM，这是一种用于区分人工智能生成文本和人类编写文本的零镜头检测方法。该方法为每个输入构建理想的人工智能生成序列，迭代地修复非最优令牌，并将累积修复工作量化为可解释的检测信号。经验评估表明，我们的方法实现了最先进的检测性能，并对各种对抗攻击和输入长度表现出强大的鲁棒性。具体来说，在多个公共基准数据集中，DNA-DetectLLM在AUROC和F1评分上相对提高了5.55%，在F1评分上相对提高了2.08%。



