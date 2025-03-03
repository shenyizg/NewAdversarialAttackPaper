# Latest Adversarial Attack Papers
**update at 2025-03-03 09:52:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions**

数据集蒸馏的演变：迈向可扩展和可推广的解决方案 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.05673v2) [paper-pdf](http://arxiv.org/pdf/2502.05673v2)

**Authors**: Ping Liu, Jiawei Du

**Abstract**: Dataset distillation, which condenses large-scale datasets into compact synthetic representations, has emerged as a critical solution for training modern deep learning models efficiently. While prior surveys focus on developments before 2023, this work comprehensively reviews recent advances, emphasizing scalability to large-scale datasets such as ImageNet-1K and ImageNet-21K. We categorize progress into a few key methodologies: trajectory matching, gradient matching, distribution matching, scalable generative approaches, and decoupling optimization mechanisms. As a comprehensive examination of recent dataset distillation advances, this survey highlights breakthrough innovations: the SRe2L framework for efficient and effective condensation, soft label strategies that significantly enhance model accuracy, and lossless distillation techniques that maximize compression while maintaining performance. Beyond these methodological advancements, we address critical challenges, including robustness against adversarial and backdoor attacks, effective handling of non-IID data distributions. Additionally, we explore emerging applications in video and audio processing, multi-modal learning, medical imaging, and scientific computing, highlighting its domain versatility. By offering extensive performance comparisons and actionable research directions, this survey equips researchers and practitioners with practical insights to advance efficient and generalizable dataset distillation, paving the way for future innovations.

摘要: 数据集蒸馏将大规模数据集浓缩为紧凑的综合表示，已成为有效训练现代深度学习模型的关键解决方案。虽然以前的调查侧重于2023年之前的发展，但这项工作全面回顾了最近的进展，强调了对大规模数据集的可扩展性，如ImageNet-1K和ImageNet-21K。我们将进展归类为几种关键方法：轨迹匹配、梯度匹配、分布匹配、可伸缩的产生式方法和解耦优化机制。随着对最新数据集蒸馏进展的全面研究，本调查重点介绍了突破性创新：用于高效和有效缩合的SRe2L框架、显著提高模型精度的软标签策略，以及在保持性能的同时最大化压缩的无损蒸馏技术。除了这些方法上的进步，我们还解决了关键挑战，包括对对手和后门攻击的健壮性，对非IID数据分发的有效处理。此外，我们还探讨了视频和音频处理、多模式学习、医学成像和科学计算中的新兴应用，突出了其领域的多功能性。通过提供广泛的性能比较和可操作的研究方向，此调查为研究人员和从业者提供了实用的见解，以推进高效和可推广的数据集蒸馏，为未来的创新铺平道路。



## **2. On Adversarial Attacks In Acoustic Drone Localization**

声学无人机定位中的对抗攻击 cs.SD

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20325v1) [paper-pdf](http://arxiv.org/pdf/2502.20325v1)

**Authors**: Tamir Shor, Chaim Baskin, Alex Bronstein

**Abstract**: Multi-rotor aerial autonomous vehicles (MAVs, more widely known as "drones") have been generating increased interest in recent years due to their growing applicability in a vast and diverse range of fields (e.g., agriculture, commercial delivery, search and rescue). The sensitivity of visual-based methods to lighting conditions and occlusions had prompted growing study of navigation reliant on other modalities, such as acoustic sensing. A major concern in using drones in scale for tasks in non-controlled environments is the potential threat of adversarial attacks over their navigational systems, exposing users to mission-critical failures, security breaches, and compromised safety outcomes that can endanger operators and bystanders. While previous work shows impressive progress in acoustic-based drone localization, prior research in adversarial attacks over drone navigation only addresses visual sensing-based systems. In this work, we aim to compensate for this gap by supplying a comprehensive analysis of the effect of PGD adversarial attacks over acoustic drone localization. We furthermore develop an algorithm for adversarial perturbation recovery, capable of markedly diminishing the affect of such attacks in our setting. The code for reproducing all experiments will be released upon publication.

摘要: 近年来，多旋翼无人驾驶飞行器(MAV，更广泛地被称为“无人机”)由于其在广泛和多样化的领域(例如，农业、商业递送、搜索和救援)的日益广泛的适用性而引起了越来越多的兴趣。基于视觉的方法对照明条件和遮挡的敏感性促使越来越多的研究依赖于其他方式的导航，如声学传感。在非受控环境中大规模使用无人机执行任务的一个主要问题是，其导航系统受到对抗性攻击的潜在威胁，使用户面临任务关键故障、安全漏洞和可能危及操作员和旁观者的安全结果。虽然以前的工作显示在基于声学的无人机定位方面取得了令人印象深刻的进展，但以前对无人机导航的对抗性攻击的研究只涉及基于视觉传感的系统。在这项工作中，我们旨在通过全面分析PGD对抗性攻击对声学无人机定位的影响来弥补这一差距。此外，我们还开发了一种对抗性扰动恢复算法，能够显著降低此类攻击在我们的环境中的影响。复制所有实验的代码将在发布时发布。



## **3. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

Logicbreaks：理解基于规则的推理颠覆的框架 cs.AI

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2407.00075v4) [paper-pdf](http://arxiv.org/pdf/2407.00075v4)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.

摘要: 我们研究如何根据预算指定的规则颠覆大型语言模型（LLM）。我们首先将规则遵循形式化为命题Horn逻辑中的推理，这是一个数学系统，其中规则的形式为“如果$P$和$Q$，那么$R$”，对于某些命题$P$、$Q$和$R$。接下来，我们证明，尽管小型变压器可以忠实地遵循这些规则，但恶意制作的提示仍然会误导理论构建和从数据中学习的模型。此外，我们证明了LLM上的流行攻击算法可以找到对抗提示并诱导与我们的理论一致的注意力模式。我们新颖的基于逻辑的框架为在基于规则的环境中研究LLM提供了基础，从而能够对逻辑推理和越狱攻击等任务进行正式分析。



## **4. Adversarial Robustness in Parameter-Space Classifiers**

参数空间分类器中的对抗鲁棒性 cs.LG

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20314v1) [paper-pdf](http://arxiv.org/pdf/2502.20314v1)

**Authors**: Tamir Shor, Ethan Fetaya, Chaim Baskin, Alex Bronstein

**Abstract**: Implicit Neural Representations (INRs) have been recently garnering increasing interest in various research fields, mainly due to their ability to represent large, complex data in a compact and continuous manner. Past work further showed that numerous popular downstream tasks can be performed directly in the INR parameter-space. Doing so can substantially reduce the computational resources required to process the represented data in their native domain. A major difficulty in using modern machine-learning approaches, is their high susceptibility to adversarial attacks, which have been shown to greatly limit the reliability and applicability of such methods in a wide range of settings. In this work, we show that parameter-space models trained for classification are inherently robust to adversarial attacks -- without the need of any robust training. To support our claims, we develop a novel suite of adversarial attacks targeting parameter-space classifiers, and furthermore analyze practical considerations of attacking parameter-space classifiers. Code for reproducing all experiments and implementation of all proposed methods will be released upon publication.

摘要: 隐式神经表示(INR)最近在各个研究领域引起了越来越多的兴趣，这主要是因为它们能够以紧凑和连续的方式表示大型、复杂的数据。过去的工作进一步表明，许多流行的下游任务可以直接在INR参数空间中执行。这样做可以大大减少在其本地域中处理所表示的数据所需的计算资源。使用现代机器学习方法的一个主要困难是它们对对手攻击的高度敏感性，这已被证明在广泛的环境中极大地限制了这种方法的可靠性和适用性。在这项工作中，我们证明了为分类而训练的参数空间模型在本质上对对手攻击是健壮的--而不需要任何健壮的训练。为了支持我们的观点，我们开发了一套新的针对参数空间分类器的对抗性攻击，并进一步分析了攻击参数空间分类器的实际考虑。复制所有实验和所有建议方法的实现的代码将在发布时发布。



## **5. SecureGaze: Defending Gaze Estimation Against Backdoor Attacks**

SecureGaze：保护Gaze估计免受后门攻击 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20306v1) [paper-pdf](http://arxiv.org/pdf/2502.20306v1)

**Authors**: Lingyu Du, Yupei Liu, Jinyuan Jia, Guohao Lan

**Abstract**: Gaze estimation models are widely used in applications such as driver attention monitoring and human-computer interaction. While many methods for gaze estimation exist, they rely heavily on data-hungry deep learning to achieve high performance. This reliance often forces practitioners to harvest training data from unverified public datasets, outsource model training, or rely on pre-trained models. However, such practices expose gaze estimation models to backdoor attacks. In such attacks, adversaries inject backdoor triggers by poisoning the training data, creating a backdoor vulnerability: the model performs normally with benign inputs, but produces manipulated gaze directions when a specific trigger is present. This compromises the security of many gaze-based applications, such as causing the model to fail in tracking the driver's attention. To date, there is no defense that addresses backdoor attacks on gaze estimation models. In response, we introduce SecureGaze, the first solution designed to protect gaze estimation models from such attacks. Unlike classification models, defending gaze estimation poses unique challenges due to its continuous output space and globally activated backdoor behavior. By identifying distinctive characteristics of backdoored gaze estimation models, we develop a novel and effective approach to reverse-engineer the trigger function for reliable backdoor detection. Extensive evaluations in both digital and physical worlds demonstrate that SecureGaze effectively counters a range of backdoor attacks and outperforms seven state-of-the-art defenses adapted from classification models.

摘要: 视线估计模型在驾驶员注意力监控、人机交互等领域有着广泛的应用。虽然存在许多凝视估计方法，但它们在很大程度上依赖于需要数据的深度学习来实现高性能。这种依赖经常迫使从业者从未经验证的公共数据集中获取培训数据，外包模型培训，或者依赖预先培训的模型。然而，这种做法使凝视估计模型面临后门攻击。在此类攻击中，攻击者通过毒化训练数据来注入后门触发器，从而产生一个后门漏洞：该模型在良性输入的情况下正常运行，但当存在特定触发器时会产生受操纵的凝视方向。这损害了许多基于凝视的应用程序的安全性，例如导致该模型无法跟踪驾驶员的注意力。到目前为止，还没有针对凝视估计模型的后门攻击的防御措施。作为回应，我们引入了SecureGaze，这是第一个旨在保护凝视估计模型免受此类攻击的解决方案。与分类模型不同，防御凝视估计由于其连续的输出空间和全局激活的后门行为而构成了独特的挑战。通过识别后门视线估计模型的显著特征，我们开发了一种新颖而有效的方法来反向工程触发函数以实现可靠的后门检测。在数字和物理世界的广泛评估表明，SecureGaze有效地对抗了一系列后门攻击，并超过了七种根据分类模型改编的最先进的防御措施。



## **6. Voting Scheme to Strengthen Localization Security in Randomly Deployed Wireless Sensor Networks**

增强随机部署无线传感器网络本地化安全性的投票方案 eess.SP

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20218v1) [paper-pdf](http://arxiv.org/pdf/2502.20218v1)

**Authors**: Slavisa Tomic, Marko Beko, Dejan Vukobratovic, Srdjan Krco

**Abstract**: This work aspires to provide a trustworthy solution for target localization in adverse environments, where malicious nodes, capable of manipulating distance measurements (i.e., performing spoofing attacks), are present, thus hindering accurate localization. Besides localization, its other goal is to identify (detect) which of the nodes participating in the process are malicious. This problem becomes extremely important with the forthcoming expansion of IoT and smart cities applications, that depend on accurate localization, and the presence of malicious attackers can represent serious security threats if not taken into consideration. This is the case with most existing localization systems which makes them highly vulnerable to spoofing attacks. In addition, existing methods that are intended for adversarial settings consider very specific settings or require additional knowledge about the system model, making them only partially secure. Therefore, this work proposes a novel voting scheme based on clustering and weighted central mass to securely solve the localization problem and detect attackers. The proposed solution has two main phases: 1) Choosing a cluster of suitable points of interest by taking advantage of the problem geometry to assigning votes in order to localize the target, and 2) Attacker detection by exploiting the location estimate and basic statistics. The proposed method is assessed in terms of localization accuracy, success in attacker detection, and computational complexity in different settings. Computer simulations and real-world experiments corroborate the effectiveness of the proposed scheme compared to state-of-the-art methods, showing that it can accomplish an error reduction of $30~\%$ and is capable of achieving almost perfect attacker detection rate when the ratio between attacker intensity and noise standard deviation is significant.

摘要: 这项工作旨在为恶劣环境中的目标定位提供可靠的解决方案，在恶劣环境中，存在能够操纵距离测量(即执行欺骗攻击)的恶意节点，从而阻碍准确的定位。除了本地化，它的另一个目标是识别(检测)参与进程的哪些节点是恶意的。随着物联网和智慧城市应用的即将扩展，这个问题变得极其重要，这些应用依赖于准确的定位，如果不考虑恶意攻击者的存在，可能会构成严重的安全威胁。大多数现有的本地化系统都是这种情况，这使得它们非常容易受到欺骗攻击。此外，用于对抗性设置的现有方法考虑非常具体的设置，或者需要关于系统模型的额外知识，使得它们只有部分安全。因此，本文提出了一种新颖的基于聚类和加权中心质量的投票方案，以安全地解决定位问题和检测攻击者。该方法主要分为两个阶段：1)利用问题的几何特征选择合适的兴趣点进行投票，以定位目标；2)利用位置估计和基本统计信息进行攻击者检测。在不同的环境下，从定位精度、攻击者检测的成功率和计算复杂度三个方面对该方法进行了评估。计算机仿真和真实世界实验验证了该方法的有效性，结果表明，与现有方法相比，该方法的误码率降低了30~5%，并且在攻击强度与噪声标准差之比较大的情况下，能够获得近乎完美的攻击检测率。



## **7. Modern DDoS Threats and Countermeasures: Insights into Emerging Attacks and Detection Strategies**

现代分布式拒绝服务威胁和对策：对新兴攻击和检测策略的见解 cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19996v1) [paper-pdf](http://arxiv.org/pdf/2502.19996v1)

**Authors**: Jincheng Wang, Le Yu, John C. S. Lui, Xiapu Luo

**Abstract**: Distributed Denial of Service (DDoS) attacks persist as significant threats to online services and infrastructure, evolving rapidly in sophistication and eluding traditional detection mechanisms. This evolution demands a comprehensive examination of current trends in DDoS attacks and the efficacy of modern detection strategies. This paper offers an comprehensive survey of emerging DDoS attacks and detection strategies over the past decade. We delve into the diversification of attack targets, extending beyond conventional web services to include newer network protocols and systems, and the adoption of advanced adversarial tactics. Additionally, we review current detection techniques, highlighting essential features that modern systems must integrate to effectively neutralize these evolving threats. Given the technological demands of contemporary network systems, such as high-volume and in-line packet processing capabilities, we also explore how innovative hardware technologies like programmable switches can significantly enhance the development and deployment of robust DDoS detection systems. We conclude by identifying open problems and proposing future directions for DDoS research. In particular, our survey sheds light on the investigation of DDoS attack surfaces for emerging systems, protocols, and adversarial strategies. Moreover, we outlines critical open questions in the development of effective detection systems, e.g., the creation of defense mechanisms independent of control planes.

摘要: 分布式拒绝服务(DDoS)攻击仍然是对在线服务和基础设施的重大威胁，其复杂性发展迅速，并绕过了传统的检测机制。这一演变需要对DDoS攻击的当前趋势和现代检测策略的有效性进行全面检查。本文对过去十年中出现的DDoS攻击和检测策略进行了全面的调查。我们深入研究了攻击目标的多样化，从传统的Web服务扩展到包括更新的网络协议和系统，以及采用先进的对抗性战术。此外，我们还回顾了当前的检测技术，重点介绍了现代系统必须集成的基本功能，以有效地中和这些不断演变的威胁。考虑到现代网络系统的技术需求，例如大容量和内联数据包处理能力，我们还将探讨可编程交换机等创新硬件技术如何显著增强强大的DDoS检测系统的开发和部署。最后，我们确定了有待解决的问题，并提出了DDoS研究的未来方向。特别是，我们的调查揭示了对新兴系统、协议和对抗策略的DDoS攻击面的调查。此外，我们概述了在开发有效的检测系统中的关键开放问题，例如，创建独立于控制平面的防御机制。



## **8. Trustworthy AI-Generative Content for Intelligent Network Service: Robustness, Security, and Fairness**

用于智能网络服务的值得信赖的人工智能生成内容：稳健性、安全性和公平性 cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2405.05930v2) [paper-pdf](http://arxiv.org/pdf/2405.05930v2)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Xiang Chen, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have revolutionized content creation. High-speed next-generation communication technology is an ideal platform for providing powerful AIGC network services. At the same time, advanced AIGC techniques can also make future network services more intelligent, especially various online content generation services. However, the significant untrustworthiness concerns of current AIGC models, such as robustness, security, and fairness, greatly affect the credibility of intelligent network services, especially in ensuring secure AIGC services. This paper proposes TrustGAIN, a trustworthy AIGC framework that incorporates robust, secure, and fair network services. We first discuss the robustness to adversarial attacks faced by AIGC models in network systems and the corresponding protection issues. Subsequently, we emphasize the importance of avoiding unsafe and illegal services and ensuring the fairness of the AIGC network services. Then as a case study, we propose a novel sentiment analysis-based detection method to guide the robust detection of unsafe content in network services. We conduct our experiments on fake news, malicious code, and unsafe review datasets to represent LLM application scenarios. Our results indicate that TrustGAIN is an exploration of future networks that can support trustworthy AIGC network services.

摘要: 以大型语言模型(LLM)为代表的AI生成内容(AIGC)模型彻底改变了内容创作。高速下一代通信技术是提供强大AIGC网络服务的理想平台。同时，先进的AIGC技术也可以让未来的网络服务更加智能化，特别是各种在线内容生成服务。然而，当前AIGC模型的健壮性、安全性和公平性等严重的不可信赖性问题极大地影响了智能网服务的可信度，特别是在确保安全的AIGC服务方面。提出了一种融合了健壮、安全、公平的网络服务的可信AIGC框架TrustGAIN。我们首先讨论了AIGC模型在网络系统中所面临的对抗攻击的稳健性以及相应的保护问题。随后，我们强调了避免不安全和非法服务的重要性，确保AIGC网络服务的公平性。作为一个案例，我们提出了一种新的基于情感分析的检测方法来指导对网络服务中不安全内容的稳健检测。我们在假新闻、恶意代码和不安全评论数据集上进行了实验，以表示LLM应用场景。我们的结果表明，TrustGAIN是对未来网络的探索，可以支持可信的AIGC网络服务。



## **9. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12659v3) [paper-pdf](http://arxiv.org/pdf/2502.12659v3)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: 大型推理模型的快速发展，如OpenAI-03和DeepSeek-R1，使得复杂推理相对于非推理的大型语言模型有了显著的改进。然而，它们增强的能力，再加上DeepSeek-R1等型号的开源访问，引发了严重的安全问题，特别是它们可能被滥用的问题。在这项工作中，我们提出了这些推理模型的全面安全评估，利用已建立的安全基准来评估它们是否符合安全法规。此外，我们调查了它们对敌意攻击的敏感性，例如越狱和快速注入，以评估它们在现实世界应用程序中的健壮性。通过多方面的分析，我们发现了四个重要的发现：(1)无论是在安全基准上还是在攻击上，开源的R1型号和03-mini型号之间都存在着显著的安全差距，这表明需要在R1上做出更多的安全努力。(2)与安全对齐的基本模型相比，精炼推理模型的安全性能较差。(3)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(4)与最终答案相比，R1模型的思维过程带来了更大的安全顾虑。我们的研究为推理模型的安全含义提供了见解，并强调了在R1模型的安全性方面进一步改进的必要性，以缩小差距。



## **10. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

一脚踏进门：LLC的多次越狱 cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.19820v2) [paper-pdf](http://arxiv.org/pdf/2502.19820v2)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.

摘要: 随着大型语言模型越来越多地融入现实世界的应用程序中，确保人工智能的安全至关重要。一个关键的挑战是越狱，敌意提示绕过内置的保护措施，导致有害的不允许输出。受心理学进门原则的启发，我们引入了FITD，这是一种新颖的多转弯越狱方法，它利用了这样一种现象，即较小的初始承诺降低了对更重大或更不道德的违法行为的抵抗力。我们的方法通过中间桥提示逐步升级用户查询的恶意意图，并使模型本身的响应保持一致，以诱导有毒响应。在两个越狱基准上的广泛实验结果表明，FITD在七个广泛使用的模型上实现了94%的平均攻击成功率，性能优于现有的最先进方法。此外，我们还提供了对LLM自我腐败的深入分析，强调了当前调整策略中的漏洞，并强调了多轮交互中固有的风险。代码可在https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.上获得



## **11. Hyperparameters in Score-Based Membership Inference Attacks**

基于分数的成员推断攻击中的超参数 cs.LG

This work has been accepted for publication in the 3rd IEEE  Conference on Secure and Trustworthy Machine Learning (SaTML'25). The final  version will be available on IEEE Xplore

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.06374v2) [paper-pdf](http://arxiv.org/pdf/2502.06374v2)

**Authors**: Gauri Pradhan, Joonas Jälkö, Marlon Tobaben, Antti Honkela

**Abstract**: Membership Inference Attacks (MIAs) have emerged as a valuable framework for evaluating privacy leakage by machine learning models. Score-based MIAs are distinguished, in particular, by their ability to exploit the confidence scores that the model generates for particular inputs. Existing score-based MIAs implicitly assume that the adversary has access to the target model's hyperparameters, which can be used to train the shadow models for the attack. In this work, we demonstrate that the knowledge of target hyperparameters is not a prerequisite for MIA in the transfer learning setting. Based on this, we propose a novel approach to select the hyperparameters for training the shadow models for MIA when the attacker has no prior knowledge about them by matching the output distributions of target and shadow models. We demonstrate that using the new approach yields hyperparameters that lead to an attack near indistinguishable in performance from an attack that uses target hyperparameters to train the shadow models. Furthermore, we study the empirical privacy risk of unaccounted use of training data for hyperparameter optimization (HPO) in differentially private (DP) transfer learning. We find no statistically significant evidence that performing HPO using training data would increase vulnerability to MIA.

摘要: 成员关系推理攻击(MIA)已经成为机器学习模型评估隐私泄露的一个有价值的框架。基于分数的MIA的特别之处在于，它们能够利用模型为特定输入生成的置信度分数。现有的基于分数的MIA隐含地假设对手可以访问目标模型的超参数，这些超参数可以用于训练攻击的影子模型。在这项工作中，我们证明了在迁移学习环境下，目标超参数的知识不是MIA的先决条件。基于此，我们提出了一种新的方法，当攻击者对阴影模型一无所知时，通过匹配目标和阴影模型的输出分布来选择用于训练MIA阴影模型的超参数。我们证明，使用新方法产生的超参数导致的攻击在性能上与使用目标超参数来训练阴影模型的攻击几乎没有区别。此外，我们还研究了在差异私有(DP)迁移学习中使用训练数据进行超参数优化(HPO)时的经验隐私风险。我们没有发现有统计学意义的证据表明，使用训练数据执行HPO会增加MIA的易感性。



## **12. Snowball Adversarial Attack on Traffic Sign Classification**

交通标志分类的滚雪球对抗攻击 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19757v1) [paper-pdf](http://arxiv.org/pdf/2502.19757v1)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial attacks on machine learning models often rely on small, imperceptible perturbations to mislead classifiers. Such strategy focuses on minimizing the visual perturbation for humans so they are not confused, and also maximizing the misclassification for machine learning algorithms. An orthogonal strategy for adversarial attacks is to create perturbations that are clearly visible but do not confuse humans, yet still maximize misclassification for machine learning algorithms. This work follows the later strategy, and demonstrates instance of it through the Snowball Adversarial Attack in the context of traffic sign recognition. The attack leverages the human brain's superior ability to recognize objects despite various occlusions, while machine learning algorithms are easily confused. The evaluation shows that the Snowball Adversarial Attack is robust across various images and is able to confuse state-of-the-art traffic sign recognition algorithm. The findings reveal that Snowball Adversarial Attack can significantly degrade model performance with minimal effort, raising important concerns about the vulnerabilities of deep neural networks and highlighting the necessity for improved defenses for image recognition machine learning models.

摘要: 对机器学习模型的对抗性攻击通常依赖于微小的、不可察觉的扰动来误导分类器。这样的策略侧重于最小化人类的视觉扰动，使他们不会感到困惑，同时也最大化了机器学习算法的误分类。对抗性攻击的一种正交策略是，创建清晰可见但不会让人类感到困惑的扰动，同时仍最大限度地提高机器学习算法的错误分类。这项工作遵循后一种策略，并通过Snowball对抗性攻击在交通标志识别的背景下演示了该策略的实例。这种攻击利用了人脑在各种遮挡情况下识别物体的优越能力，而机器学习算法很容易混淆。评估表明，Snowball对抗性攻击在各种图像上都是健壮的，并且能够混淆最先进的交通标志识别算法。研究结果表明，Snowball恶意攻击能够以最小的努力显著降低模型的性能，这引发了人们对深度神经网络脆弱性的重要担忧，并突显了改进图像识别机器学习模型防御的必要性。



## **13. HALO: Robust Out-of-Distribution Detection via Joint Optimisation**

HALO：通过联合优化实现稳健的分布外检测 cs.LG

SaTML 2025

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19755v1) [paper-pdf](http://arxiv.org/pdf/2502.19755v1)

**Authors**: Hugo Lyons Keenan, Sarah Erfani, Christopher Leckie

**Abstract**: Effective out-of-distribution (OOD) detection is crucial for the safe deployment of machine learning models in real-world scenarios. However, recent work has shown that OOD detection methods are vulnerable to adversarial attacks, potentially leading to critical failures in high-stakes applications. This discovery has motivated work on robust OOD detection methods that are capable of maintaining performance under various attack settings. Prior approaches have made progress on this problem but face a number of limitations: often only exhibiting robustness to attacks on OOD data or failing to maintain strong clean performance. In this work, we adapt an existing robust classification framework, TRADES, extending it to the problem of robust OOD detection and discovering a novel objective function. Recognising the critical importance of a strong clean/robust trade-off for OOD detection, we introduce an additional loss term which boosts classification and detection performance. Our approach, called HALO (Helper-based AdversariaL OOD detection), surpasses existing methods and achieves state-of-the-art performance across a number of datasets and attack settings. Extensive experiments demonstrate an average AUROC improvement of 3.15 in clean settings and 7.07 under adversarial attacks when compared to the next best method. Furthermore, HALO exhibits resistance to transferred attacks, offers tuneable performance through hyperparameter selection, and is compatible with existing OOD detection frameworks out-of-the-box, leaving open the possibility of future performance gains. Code is available at: https://github.com/hugo0076/HALO

摘要: 有效的分布外(OOD)检测对于机器学习模型在真实场景中的安全部署至关重要。然而，最近的工作表明，OOD检测方法容易受到对手攻击，可能会导致高风险应用程序中的关键故障。这一发现推动了健壮的OOD检测方法的研究，这些方法能够在各种攻击设置下保持性能。以前的方法已经在这个问题上取得了进展，但面临着一些限制：通常只对OOD数据的攻击表现出健壮性，或者无法保持强大的干净性能。在这项工作中，我们采用了一个现有的稳健分类框架TRADS，将其扩展到稳健的OOD检测问题，并发现了一个新的目标函数。认识到强大的干净/健壮的权衡对于OOD检测的关键重要性，我们引入了一个额外的损失项来提高分类和检测性能。我们的方法被称为HALO(基于Helper-Based的对抗性OOD检测)，它超越了现有的方法，并在许多数据集和攻击设置上获得了最先进的性能。广泛的实验表明，与次好的方法相比，干净环境下的AUROC平均提高了3.15，而在对抗性攻击下的AUROC平均提高了7.07。此外，Halo表现出对转移攻击的抵抗力，通过超参数选择提供可调的性能，并与现有的开箱即用的OOD检测框架兼容，留下了未来性能提升的可能性。代码可从以下网址获得：https://github.com/hugo0076/HALO



## **14. SAP-DIFF: Semantic Adversarial Patch Generation for Black-Box Face Recognition Models via Diffusion Models**

SAP-Diff：通过扩散模型为黑匣子人脸识别模型生成语义对抗补丁 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19710v1) [paper-pdf](http://arxiv.org/pdf/2502.19710v1)

**Authors**: Mingsi Wang, Shuaiyin Yao, Chang Yue, Lijie Zhang, Guozhu Meng

**Abstract**: Given the need to evaluate the robustness of face recognition (FR) models, many efforts have focused on adversarial patch attacks that mislead FR models by introducing localized perturbations. Impersonation attacks are a significant threat because adversarial perturbations allow attackers to disguise themselves as legitimate users. This can lead to severe consequences, including data breaches, system damage, and misuse of resources. However, research on such attacks in FR remains limited. Existing adversarial patch generation methods exhibit limited efficacy in impersonation attacks due to (1) the need for high attacker capabilities, (2) low attack success rates, and (3) excessive query requirements. To address these challenges, we propose a novel method SAP-DIFF that leverages diffusion models to generate adversarial patches via semantic perturbations in the latent space rather than direct pixel manipulation. We introduce an attention disruption mechanism to generate features unrelated to the original face, facilitating the creation of adversarial samples and a directional loss function to guide perturbations toward the target identity feature space, thereby enhancing attack effectiveness and efficiency. Extensive experiments on popular FR models and datasets demonstrate that our method outperforms state-of-the-art approaches, achieving an average attack success rate improvement of 45.66% (all exceeding 40%), and a reduction in the number of queries by about 40% compared to the SOTA approach

摘要: 由于需要评估人脸识别(FR)模型的稳健性，许多努力都集中在通过引入局部扰动来误导FR模型的对抗性补丁攻击。模仿攻击是一个重大威胁，因为敌意扰动允许攻击者伪装成合法用户。这可能会导致严重的后果，包括数据泄露、系统损坏和资源滥用。然而，对FR中此类攻击的研究仍然有限。现有的对抗性补丁生成方法在模拟攻击中的效果有限，原因是(1)需要高攻击能力，(2)攻击成功率低，(3)查询要求过高。为了应对这些挑战，我们提出了一种新的方法SAP-DIFF，该方法利用扩散模型通过潜在空间中的语义扰动而不是直接的像素操作来生成敌意补丁。引入注意力干扰机制来生成与原始人脸无关的特征，便于创建敌方样本，并引入方向损失函数来引导扰动向目标身份特征空间移动，从而提高攻击的有效性和效率。在流行的FR模型和数据集上的大量实验表明，该方法的性能优于最新的方法，平均攻击成功率提高了45.66%(均超过40%)，查询次数比SOTA方法减少了约40%



## **15. LiD-FL: Towards List-Decodable Federated Learning**

LiD-FL：迈向列表可解码联邦学习 cs.LG

26 pages, 5 figures

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2408.04963v3) [paper-pdf](http://arxiv.org/pdf/2408.04963v3)

**Authors**: Hong Liu, Liren Shan, Han Bao, Ronghui You, Yuhao Yi, Jiancheng Lv

**Abstract**: Federated learning is often used in environments with many unverified participants. Therefore, federated learning under adversarial attacks receives significant attention. This paper proposes an algorithmic framework for list-decodable federated learning, where a central server maintains a list of models, with at least one guaranteed to perform well. The framework has no strict restriction on the fraction of honest workers, extending the applicability of Byzantine federated learning to the scenario with more than half adversaries. Under proper assumptions on the loss function, we prove a convergence theorem for our method. Experimental results, including image classification tasks with both convex and non-convex losses, demonstrate that the proposed algorithm can withstand the malicious majority under various attacks.

摘要: 联邦学习通常用于有许多未经验证的参与者的环境中。因此，对抗攻击下的联邦学习受到了高度关注。本文提出了一种用于列表可解码联邦学习的算法框架，其中中央服务器维护一个模型列表，至少有一个模型保证表现良好。该框架对诚实工人的比例没有严格限制，将拜占庭联邦学习的适用性扩展到有一半以上对手的场景。在对损失函数的适当假设下，我们证明了我们方法的收敛性定理。实验结果（包括具有凸损失和非凸损失的图像分类任务）表明，所提出的算法可以抵御各种攻击下的恶意多数。



## **16. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

基于属性感知文本倒置的预算驱动可转移对抗攻击 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19697v1) [paper-pdf](http://arxiv.org/pdf/2502.19697v1)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.

摘要: 人员再识别(re-id)模型在安全监控系统中是至关重要的，需要可转移的对抗性攻击来探索其脆弱性。近年来，基于视觉语言模型(VLM)的攻击通过攻击VLM的广义图像和文本特征表现出了良好的可转移性，但由于过于强调积分表示的区分语义，缺乏全面的特征破坏。本文介绍了一种新的基于属性感知的提示攻击(AP-Attack)，该方法利用VLM的图文对齐能力，通过破坏特定于属性的文本嵌入来显式破坏行人图像的细粒度语义特征。为了获得个性化的属性文本描述，文本倒置网络被设计成将行人图像映射到代表语义嵌入的伪标记，并利用图像和预定义的提示模板进行对比学习，以明确描述行人属性。反转良性和对抗性细粒度的文本语义有助于攻击者有效地进行彻底的破坏，增强了对抗性实例的可转移性。大量实验表明，AP-Attack具有最好的可转移性，在跨模型和数据集攻击场景下，平均丢失率比以前的方法高出22.9%。



## **17. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

MAA：针对视觉语言预训练模型的强力对抗攻击 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.08079v2) [paper-pdf](http://arxiv.org/pdf/2502.08079v2)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.

摘要: 当前用于评估视觉语言预训练(VLP)模型在多模式任务中的稳健性的对抗性攻击存在可转移性有限的问题，其中针对特定模型的攻击往往难以在不同的模型上有效地泛化，从而限制了它们在更广泛地评估稳健性方面的有效性。这主要归因于过度依赖特定型号的特征和区域，特别是在图像模式方面。在本文中，我们提出了一种优雅而高效的方法，称为精细攻击(MAA)，它充分利用了个体样本的模型无关特性和脆弱性，从而增强了泛化能力，降低了模型依赖。MAA通过开发一种新的调整大小和滑动裁剪(RSCrop)技术，结合多粒度相似破坏(MGSD)策略，强调对抗性图像的细粒度优化。在不同的VLP模型、多个基准数据集和各种下游任务上的广泛实验表明，MAA显著增强了对抗性攻击的有效性和可转移性。我们进行了大量的性能研究，以深入了解各种型号配置的有效性，从而指导该领域的未来发展。



## **18. Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack**

通过动态视觉语言对齐攻击提高MLLM中的对抗可移植性 cs.CV

arXiv admin note: text overlap with arXiv:2403.09766

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19672v1) [paper-pdf](http://arxiv.org/pdf/2502.19672v1)

**Authors**: Chenhe Gu, Jindong Gu, Andong Hua, Yao Qin

**Abstract**: Multimodal Large Language Models (MLLMs), built upon LLMs, have recently gained attention for their capabilities in image recognition and understanding. However, while MLLMs are vulnerable to adversarial attacks, the transferability of these attacks across different models remains limited, especially under targeted attack setting. Existing methods primarily focus on vision-specific perturbations but struggle with the complex nature of vision-language modality alignment. In this work, we introduce the Dynamic Vision-Language Alignment (DynVLA) Attack, a novel approach that injects dynamic perturbations into the vision-language connector to enhance generalization across diverse vision-language alignment of different models. Our experimental results show that DynVLA significantly improves the transferability of adversarial examples across various MLLMs, including BLIP2, InstructBLIP, MiniGPT4, LLaVA, and closed-source models such as Gemini.

摘要: 基于LLM构建的多模式大型语言模型（MLLM）最近因其在图像识别和理解方面的能力而受到关注。然而，虽然MLLM容易受到对抗攻击，但这些攻击在不同模型之间的可转移性仍然有限，尤其是在有针对性的攻击环境下。现有的方法主要关注特定于视觉的干扰，但难以应对视觉语言模式对齐的复杂性质。在这项工作中，我们引入了动态视觉语言对齐（DynVLA）攻击，这是一种新颖的方法，将动态扰动注入视觉语言连接器中，以增强不同模型的不同视觉语言对齐的概括性。我们的实验结果表明，DynVLA显着提高了对抗性示例在各种MLLM之间的可移植性，包括BLIP 2、INSTBLIP、MiniGPT 4、LLaVA和Gemini等闭源模型。



## **19. Algebraic Adversarial Attacks on Integrated Gradients**

对综合学生的代数对抗攻击 cs.LG

To appear in the proceedings of the International Conference on  Machine Learning and Cybernetics (ICMLC)

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2407.16233v2) [paper-pdf](http://arxiv.org/pdf/2407.16233v2)

**Authors**: Lachlan Simpson, Federico Costanza, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

**Abstract**: Adversarial attacks on explainability models have drastic consequences when explanations are used to understand the reasoning of neural networks in safety critical systems. Path methods are one such class of attribution methods susceptible to adversarial attacks. Adversarial learning is typically phrased as a constrained optimisation problem. In this work, we propose algebraic adversarial examples and study the conditions under which one can generate adversarial examples for integrated gradients. Algebraic adversarial examples provide a mathematically tractable approach to adversarial examples.

摘要: 当使用解释来理解安全关键系统中神经网络的推理时，对可解释性模型的对抗攻击会产生严重后果。路径方法是一类容易受到对抗攻击的归因方法。对抗学习通常被描述为一个受约束的优化问题。在这项工作中，我们提出了代数对抗性示例，并研究了可以为综合梯度生成对抗性示例的条件。代数对抗性例子为对抗性例子提供了一种数学上易于处理的方法。



## **20. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Proactive Adversarial Defense**

快速抢占：前向-后向级联学习，实现高效且可转移的主动对抗防御 cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2407.15524v6) [paper-pdf](http://arxiv.org/pdf/2407.15524v6)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning has advanced significantly but remains vulnerable to adversarial attacks, compromising its reliability. While conventional defenses typically mitigate perturbations post-attack, few studies explore proactive strategies for preemptive protection. This paper proposes Fast Preemption, a novel defense that neutralizes adversarial effects before third-party attacks. By employing distinct models for input labeling and feature extraction, Fast Preemption enables an efficient, transferable preemptive defense with state-of-the-art robustness across diverse systems. To further enhance efficiency, we introduce a forward-backward cascade learning algorithm that generates protective perturbations, leveraging forward propagation for rapid convergence and iterative backward propagation to mitigate overfitting. Executing in just three iterations, Fast Preemption surpasses existing training-time, test-time, and preemptive defenses. Additionally, we propose the first effective white-box adaptive reversion attack to evaluate the reversibility of preemptive defenses, demonstrating that our approach remains secure unless the backbone model, algorithm, and settings are entirely compromised. This work establishes a new paradigm for proactive adversarial defense.

摘要: 深度学习已经取得了显著进展，但仍然容易受到对手的攻击，从而影响了其可靠性。虽然常规防御通常会缓解攻击后的干扰，但很少有研究探索先发制人的保护策略。本文提出了一种新的防御机制--快速抢占，它可以在第三方攻击之前消除敌意。通过使用不同的输入标记和特征提取模型，Fast Preemption实现了跨不同系统的具有最先进健壮性的高效、可转移的先发制人防御。为了进一步提高效率，我们引入了一种前向-后向级联学习算法，该算法产生保护性扰动，利用前向传播来快速收敛，并利用迭代后向传播来减少过拟合。快速先发制人只需三次迭代，就超过了现有的训练时间、测试时间和先发制人的防御能力。此外，我们提出了第一个有效的白盒自适应逆转攻击来评估抢占防御的可逆性，证明了我们的方法仍然是安全的，除非主干模型、算法和设置完全受损。这项工作为主动对抗防御建立了一个新的范式。



## **21. EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models**

EMPRA：针对神经排名模型的嵌入扰动排名攻击 cs.IR

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2412.16382v2) [paper-pdf](http://arxiv.org/pdf/2412.16382v2)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: Recent research has shown that neural information retrieval techniques may be susceptible to adversarial attacks. Adversarial attacks seek to manipulate the ranking of documents, with the intention of exposing users to targeted content. In this paper, we introduce the Embedding Perturbation Rank Attack (EMPRA) method, a novel approach designed to perform adversarial attacks on black-box Neural Ranking Models (NRMs). EMPRA manipulates sentence-level embeddings, guiding them towards pertinent context related to the query while preserving semantic integrity. This process generates adversarial texts that seamlessly integrate with the original content and remain imperceptible to humans. Our extensive evaluation conducted on the widely-used MS MARCO V1 passage collection demonstrate the effectiveness of EMPRA against a wide range of state-of-the-art baselines in promoting a specific set of target documents within a given ranked results. Specifically, EMPRA successfully achieves a re-ranking of almost 96% of target documents originally ranked between 51-100 to rank within the top 10. Furthermore, EMPRA does not depend on surrogate models for adversarial text generation, enhancing its robustness against different NRMs in realistic settings.

摘要: 最近的研究表明，神经信息检索技术可能容易受到对抗性攻击。敌意攻击试图操纵文档的排名，目的是让用户接触到有针对性的内容。介绍了一种新的针对黑盒神经网络排名模型(NRM)的对抗性攻击方法--嵌入扰动等级攻击方法。EMPRA操作语句级别的嵌入，引导它们指向与查询相关的上下文，同时保持语义完整性。这一过程产生了与原始内容无缝集成的对抗性文本，并且对人类来说仍然是不可感知的。我们对广泛使用的MS Marco V1文章集进行了广泛的评估，证明了EMPRA在推广给定排名结果中的一组特定目标文档方面相对于广泛的最先进基线的有效性。具体地说，EMPRA成功地实现了几乎96%的目标文档的重新排序，这些文档最初的排名在51-100之间，进入前10名。此外，EMPRA不依赖于生成敌意文本的代理模型，增强了它在现实环境中对不同NRM的稳健性。



## **22. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

对抗非中心扰动：用$l_1$-模估计精确恢复线性系统 math.OC

8 pages, 2 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2410.03218v4) [paper-pdf](http://arxiv.org/pdf/2410.03218v4)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are arbitrary. We show that when the probability of having an attack at a given time is less than 0.5 and each attack spans the entire space in expectation, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.

摘要: 本文研究一般情况下的线性系统辨识问题，其中扰动是亚高斯的，相关的，可能是对抗性的。首先，我们考虑了具有非中心(非零均值)扰动的情况，对于这种情况，普通的最小二乘(OLS)方法不能正确地辨识系统。我们证明了在每个扰动具有相等的正负概率的条件下，$L_1$-范数估计量能够准确地辨识系统。这一条件限制了每个扰动的符号，但允许其大小任意。其次，在攻击次数偶然发生但攻击值的分布是任意的情况下，我们考虑了每次扰动都是对抗性的情况。证明了当给定时刻发生攻击的概率小于0.5时，当每次攻击跨越期望的整个空间时，$L_1$-范数估计对任何对抗性非中心扰动都是有效的，并且在有限时间内实现了精确的恢复。这些结果为在安全关键系统中有效防御任意规模的非中心攻击铺平了道路。



## **23. No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data**

不，我当然可以！使用无害微调数据可以利用拒绝机制 cs.CR

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19537v1) [paper-pdf](http://arxiv.org/pdf/2502.19537v1)

**Authors**: Joshua Kazdan, Lisa Yu, Rylan Schaeffer, Chris Cundy, Sanmi Koyejo, Dvijotham Krishnamurthy

**Abstract**: Leading language model (LM) providers like OpenAI and Google offer fine-tuning APIs that allow customers to adapt LMs for specific use cases. To prevent misuse, these LM providers implement filtering mechanisms to block harmful fine-tuning data. Consequently, adversaries seeking to produce unsafe LMs via these APIs must craft adversarial training data that are not identifiably harmful. We make three contributions in this context: 1. We show that many existing attacks that use harmless data to create unsafe LMs rely on eliminating model refusals in the first few tokens of their responses. 2. We show that such prior attacks can be blocked by a simple defense that pre-fills the first few tokens from an aligned model before letting the fine-tuned model fill in the rest. 3. We describe a new data-poisoning attack, ``No, Of course I Can Execute'' (NOICE), which exploits an LM's formulaic refusal mechanism to elicit harmful responses. By training an LM to refuse benign requests on the basis of safety before fulfilling those requests regardless, we are able to jailbreak several open-source models and a closed-source model (GPT-4o). We show an attack success rate (ASR) of 57% against GPT-4o; our attack earned a Bug Bounty from OpenAI. Against open-source models protected by simple defenses, we improve ASRs by an average of 3.25 times compared to the best performing previous attacks that use only harmless data. NOICE demonstrates the exploitability of repetitive refusal mechanisms and broadens understanding of the threats closed-source models face from harmless data.

摘要: 领先的语言模型(LM)提供商，如OpenAI和Google，提供了微调的API，允许客户根据特定的用例调整LMS。为防止误用，这些LM提供程序实施过滤机制以阻止有害的微调数据。因此，试图通过这些API生成不安全的LMS的攻击者必须创建不能识别有害的对抗性训练数据。我们在这方面做了三点贡献：1.我们证明了许多现有的使用无害数据来创建不安全的LMS的攻击依赖于消除其响应的前几个令牌中的模型拒绝。2.我们证明了这样的先前攻击可以通过一个简单的防御来阻止，即预先填充来自对齐模型的前几个令牌，然后让微调的模型填充其余的令牌。3.我们描述了一种新的数据中毒攻击，``不，我当然可以执行‘’(Noice)，它利用LM的公式化拒绝机制来引发有害的响应。通过训练LM在满足良性请求之前出于安全考虑拒绝这些请求，我们能够越狱几个开源模型和一个封闭源代码模型(GPT-40)。我们对GPT-40的攻击成功率(ASR)为57%；我们的攻击从OpenAI获得了错误赏金。相对于由简单防御保护的开源模型，我们将ASR平均提高了3.25倍，而之前仅使用无害数据的攻击性能最好。Noice展示了重复拒绝机制的可利用性，并拓宽了对封闭源代码模型面临的无害数据威胁的理解。



## **24. Unveiling Wireless Users' Locations via Modulation Classification-based Passive Attack**

通过基于调制分类的被动攻击揭露无线用户位置 cs.IT

7 pages, 4 figures, submitted to IEEE for possible publication

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19341v1) [paper-pdf](http://arxiv.org/pdf/2502.19341v1)

**Authors**: Ali Hanif, Abdulrahman Katranji, Nour Kouzayha, Muhammad Mahboob Ur Rahman, Tareq Y. Al-Naffouri

**Abstract**: The broadcast nature of the wireless medium and openness of wireless standards, e.g., 3GPP releases 16-20, invite adversaries to launch various active and passive attacks on cellular and other wireless networks. This work identifies one such loose end of wireless standards and presents a novel passive attack method enabling an eavesdropper (Eve) to localize a line of sight wireless user (Bob) who is communicating with a base station or WiFi access point (Alice). The proposed attack involves two phases. In the first phase, Eve performs modulation classification by intercepting the downlink channel between Alice and Bob. This enables Eve to utilize the publicly available modulation and coding scheme (MCS) tables to do pesudo-ranging, i.e., the Eve determines the ring within which Bob is located, which drastically reduces the search space. In the second phase, Eve sniffs the uplink channel, and employs multiple strategies to further refine Bob's location within the ring. Towards the end, we present our thoughts on how this attack can be extended to non-line-of-sight scenarios, and how this attack could act as a scaffolding to construct a malicious digital twin map.

摘要: 无线介质的广播特性和无线标准的开放性(例如，3GPP版本16-20)邀请对手对蜂窝和其他无线网络发起各种主动和被动攻击。这项工作识别了一种无线标准的松散端，并提出了一种新的被动攻击方法，使窃听者(Eve)能够定位与基站或WiFi接入点(Alice)通信的视线无线用户(Bob)。拟议的攻击包括两个阶段。在第一阶段，EVE通过截获Alice和Bob之间的下行链路信道来执行调制分类。这使得EVE能够利用公共可用的调制和编码方案(MCS)表来进行伪测距，即，EVE确定Bob所在的环，这大大减少了搜索空间。在第二阶段，Eve嗅探上行链路信道，并采用多种策略进一步确定Bob在环内的位置。最后，我们提出了我们的想法，如何将这种攻击扩展到非视线场景，以及这种攻击如何充当构建恶意数字孪生地图的脚手架。



## **25. Extreme vulnerability to intruder attacks destabilizes network dynamics**

对入侵者攻击的极端脆弱性会破坏网络动态的稳定 nlin.AO

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.08552v2) [paper-pdf](http://arxiv.org/pdf/2502.08552v2)

**Authors**: Amirhossein Nazerian, Sahand Tangerami, Malbor Asllani, David Phillips, Hernan Makse, Francesco Sorrentino

**Abstract**: Consensus, synchronization, formation control, and power grid balance are all examples of virtuous dynamical states that may arise in networks. Here we focus on how such states can be destabilized from a fundamental perspective; namely, we address the question of how one or a few intruder agents within an otherwise functioning network may compromise its dynamics. We show that a single adversarial node coupled via adversarial couplings to one or more other nodes is sufficient to destabilize the entire network, which we prove to be more efficient than targeting multiple nodes. Then, we show that concentrating the attack on a single low-indegree node induces the greatest instability, challenging the common assumption that hubs are the most critical nodes. This leads to a new characterization of the vulnerability of a node, which contrasts with previous work, and identifies low-indegree nodes (as opposed to the hubs) as the most vulnerable components of a network. Our results are derived for linear systems but hold true for nonlinear networks, including those described by the Kuramoto model. Finally, we derive scaling laws showing that larger networks are less susceptible, on average, to single-node attacks. Overall, these findings highlight an intrinsic vulnerability of technological systems such as autonomous networks, sensor networks, power grids, and the internet of things, with implications also to the realm of complex social and biological networks.

摘要: 共识、同步、队形控制和电网平衡都是网络中可能出现的良性动态状态的例子。在这里，我们从根本的角度关注如何破坏这种状态的稳定；也就是，我们解决了一个或几个入侵者代理在其他功能正常的网络中如何可能危及其动态的问题。我们证明了单个敌意节点通过对抗性耦合耦合到一个或多个其他节点足以破坏整个网络的稳定，我们证明了这比针对多个节点更有效。然后，我们证明了将攻击集中在单个低度节点上会导致最大的不稳定性，挑战了集线器是最关键节点的普遍假设。这导致了对节点脆弱性的新的表征，这与以前的工作形成了对比，并将低索引度节点(而不是集线器)识别为网络中最脆弱的组件。我们的结果适用于线性系统，但也适用于非线性网络，包括用Kuramoto模型描述的网络。最后，我们推导出了标度律，表明较大的网络平均而言不太容易受到单节点攻击。总体而言，这些发现突显了自主网络、传感器网络、电网和物联网等技术系统的内在脆弱性，这也对复杂的社会和生物网络领域产生了影响。



## **26. On the Byzantine Fault Tolerance of signSGD with Majority Vote**

论多数票签名新加坡元的拜占庭式过失容忍 cs.LG

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19170v1) [paper-pdf](http://arxiv.org/pdf/2502.19170v1)

**Authors**: Emanuele Mengoli, Luzius Moll, Virgilio Strozzi, El-Mahdi El-Mhamdi

**Abstract**: In distributed learning, sign-based compression algorithms such as signSGD with majority vote provide a lightweight alternative to SGD with an additional advantage: fault tolerance (almost) for free. However, for signSGD with majority vote, this fault tolerance has been shown to cover only the case of weaker adversaries, i.e., ones that are not omniscient or cannot collude to base their attack on common knowledge and strategy. In this work, we close this gap and provide new insights into how signSGD with majority vote can be resilient against omniscient and colluding adversaries, which craft an attack after communicating with other adversaries, thus having better information to perform the most damaging attack based on a common optimal strategy. Our core contribution is in providing a proof that begins by defining the omniscience framework and the strongest possible damage against signSGD with majority vote without imposing any restrictions on the attacker. Thanks to the filtering effect of the sign-based method, we upper-bound the space of attacks to the optimal strategy for maximizing damage by an attacker. Hence, we derive an explicit probabilistic bound in terms of incorrect aggregation without resorting to unknown constants, providing a convergence bound on signSGD with majority vote in the presence of Byzantine attackers, along with a precise convergence rate. Our findings are supported by experiments on the MNIST dataset in a distributed learning environment with adversaries of varying strength.

摘要: 在分布式学习中，基于符号的压缩算法，如多数投票的signSGD，提供了一种轻量级的SGD替代方案，具有额外的优势：几乎是免费的容错。然而，对于拥有多数票的signSGD来说，这种容错已经被证明只涵盖较弱的对手的情况，即那些不是无所不知的或不能串通以基于常识和策略的攻击的情况。在这项工作中，我们缩小了这一差距，并提供了新的见解，即拥有多数选票的signSGD如何具有抵御无所不知和串通的对手的能力，这些对手在与其他对手沟通后策划攻击，从而拥有更好的信息，基于共同的最优策略执行最具破坏性的攻击。我们的核心贡献是提供了一种证明，首先定义了无所不知的框架，并以多数票对signSGD造成了最强的破坏，而不对攻击者施加任何限制。由于基于符号的方法的过滤效果，我们将攻击空间上界到最优策略，以最大化攻击者的损害。因此，我们在不求助于未知常量的情况下，得到了不正确聚集的显式概率界，在拜占庭攻击者存在的情况下，提供了带多数投票的signSGD的收敛界，并提供了精确的收敛速度。我们的发现得到了在MNIST数据集上的实验支持，该实验在分布式学习环境中具有不同强度的对手。



## **27. XSS Adversarial Attacks Based on Deep Reinforcement Learning: A Replication and Extension Study**

基于深度强化学习的XSS对抗攻击：复制和扩展研究 cs.SE

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19095v1) [paper-pdf](http://arxiv.org/pdf/2502.19095v1)

**Authors**: Samuele Pasini, Gianluca Maragliano, Jinhan Kim, Paolo Tonella

**Abstract**: Cross-site scripting (XSS) poses a significant threat to web application security. While Deep Learning (DL) has shown remarkable success in detecting XSS attacks, it remains vulnerable to adversarial attacks due to the discontinuous nature of its input-output mapping. These adversarial attacks employ mutation-based strategies for different components of XSS attack vectors, allowing adversarial agents to iteratively select mutations to evade detection. Our work replicates a state-of-the-art XSS adversarial attack, highlighting threats to validity in the reference work and extending it toward a more effective evaluation strategy. Moreover, we introduce an XSS Oracle to mitigate these threats. The experimental results show that our approach achieves an escape rate above 96% when the threats to validity of the replicated technique are addressed.

摘要: 跨站脚本（XSS）对Web应用程序安全构成重大威胁。虽然深度学习（DL）在检测XSS攻击方面取得了巨大成功，但由于其输入输出映射的不连续性，它仍然容易受到对抗攻击。这些对抗性攻击针对XSS攻击载体的不同组成部分采用基于突变的策略，允许对抗性代理迭代地选择突变以逃避检测。我们的工作复制了最先进的XSS对抗攻击，强调了对参考工作有效性的威胁，并将其扩展到更有效的评估策略。此外，我们还引入了XSS Oracle来缓解这些威胁。实验结果表明，当解决对复制技术有效性的威胁时，我们的方法实现了96%以上的逃脱率。



## **28. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

超越表面层面模式：针对LLM越狱攻击的敏捷驱动防御框架 cs.CR

15 pages, 12 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19041v1) [paper-pdf](http://arxiv.org/pdf/2502.19041v1)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.

摘要: 尽管统一的大型语言模型(LLM)经过了拒绝有害请求的培训，但它们仍然容易受到越狱攻击。遗憾的是，现有的方法往往只关注表层模式，而忽略了更深层次的攻击本质。其结果是，当攻击提示改变时，防御就会失败，即使潜在的“攻击本质”保持不变。为了解决这个问题，我们引入了EDDF，一个针对LLMS中越狱攻击的EDDF框架。EDDF是一种即插即用的输入过滤方法，分为两个阶段：1)离线本质数据库构建，2)在线恶意查询检测。EDDF背后的关键思想是从一组不同的已知攻击实例中提取“攻击本质”，并将其存储在脱机矢量数据库中。实验结果表明，EDDF的性能明显优于现有方法，攻击成功率降低了至少20%，突出了其对越狱攻击的卓越稳健性。



## **29. Robust Over-the-Air Computation with Type-Based Multiple Access**

具有基于类型的多路访问的稳健空中计算 eess.SP

Paper submitted to 33rd European Signal Processing Conference  (EUSIPCO 2025)

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19014v1) [paper-pdf](http://arxiv.org/pdf/2502.19014v1)

**Authors**: Marc Martinez-Gost, Ana Pérez-Neira, Miguel Ángel Lagunas

**Abstract**: This paper utilizes the properties of type-based multiple access (TBMA) to investigate its effectiveness as a robust approach for over-the-air computation (AirComp) in the presence of Byzantine attacks, this is, adversarial strategies where malicious nodes intentionally distort their transmissions to corrupt the aggregated result. Unlike classical direct aggregation (DA) AirComp, which aggregates data in the amplitude of the signals and are highly vulnerable to attacks, TBMA distributes data over multiple radio resources, enabling the receiver to construct a histogram representation of the transmitted data. This structure allows the integration of classical robust estimators and supports the computation of diverse functions beyond the arithmetic mean, which is not feasible with DA. Through extensive simulations, we demonstrate that robust TBMA significantly outperforms DA, maintaining high accuracy even under adversarial conditions, and showcases its applicability in federated learning (FEEL) scenarios. Additionally, TBMA reduces channel state information (CSI) requirements, lowers energy consumption, and enhances resiliency by leveraging the diversity of the transmitted data. These results establish TBMA as a scalable and robust solution for AirComp, paving the way for secure and efficient aggregation in next-generation networks.

摘要: 利用基于类型的多路访问(TBMA)的特性，研究了在拜占庭攻击(即恶意节点故意扭曲其传输以破坏聚集结果)的情况下，其作为一种健壮的空中计算方法(AirComp)的有效性。与传统的直接聚合(DA)AirComp不同，直接聚合(DA)AirComp以信号的幅度聚合数据，并且极易受到攻击，而TBMA将数据分布在多个无线电资源上，使接收器能够构建传输数据的直方图表示。这种结构允许集成经典的稳健估计器，并支持超过算术平均值的各种函数的计算，而这在DA中是不可行的。通过大量的仿真实验，我们证明了稳健的TBMA算法明显优于DA算法，即使在对抗环境下也能保持较高的准确率，并展示了它在联合学习(Feel)场景中的适用性。此外，通过利用传输数据的分集，TBMA降低了对信道状态信息(CSI)的要求，降低了能耗，并增强了恢复能力。这些结果使TBMA成为AirComp的可扩展和强大的解决方案，为下一代网络中安全高效的聚合铺平了道路。



## **30. Learning atomic forces from uncertainty-calibrated adversarial attacks**

从不确定性校准的对抗攻击中学习原子力 physics.comp-ph

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18314v2) [paper-pdf](http://arxiv.org/pdf/2502.18314v2)

**Authors**: Henrique Musseli Cezar, Tilmann Bodenstein, Henrik Andersen Sveinsson, Morten Ledum, Simen Reine, Sigbjørn Løland Bore

**Abstract**: Adversarial approaches, which intentionally challenge machine learning models by generating difficult examples, are increasingly being adopted to improve machine learning interatomic potentials (MLIPs). While already providing great practical value, little is known about the actual prediction errors of MLIPs on adversarial structures and whether these errors can be controlled. We propose the Calibrated Adversarial Geometry Optimization (CAGO) algorithm to discover adversarial structures with user-assigned errors. Through uncertainty calibration, the estimated uncertainty of MLIPs is unified with real errors. By performing geometry optimization for calibrated uncertainty, we reach adversarial structures with the user-assigned target MLIP prediction error. Integrating with active learning pipelines, we benchmark CAGO, demonstrating stable MLIPs that systematically converge structural, dynamical, and thermodynamical properties for liquid water and water adsorption in a metal-organic framework within only hundreds of training structures, where previously many thousands were typically required.

摘要: 对抗性方法通过生成困难的例子来故意挑战机器学习模型，越来越多地被用来改善机器学习的原子间势(MLIP)。虽然已经提供了很大的实用价值，但对于MLIP在对抗性结构上的实际预测误差以及这些误差是否可以控制，人们知之甚少。我们提出了校准对抗性几何优化(CAGO)算法来发现含有用户指定错误的对抗性结构。通过不确定度校正，将MLIP的估计不确定度与实际误差统一起来。通过对标定的不确定性进行几何优化，我们得到了具有用户指定的目标MLIP预测误差的对抗性结构。与主动学习管道集成，我们对CAGO进行了基准测试，展示了稳定的MLIP，这些MLIP系统地聚合了液态水的结构、动力学和热力学属性，以及金属-有机框架中的水吸附，仅在数百个培训结构中，而以前通常需要数千个。



## **31. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models**

针对预训练的大型语言模型的纯标签成员推断攻击 cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18943v1) [paper-pdf](http://arxiv.org/pdf/2502.18943v1)

**Authors**: Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Membership Inference Attacks (MIAs) aim to predict whether a data sample belongs to the model's training set or not. Although prior research has extensively explored MIAs in Large Language Models (LLMs), they typically require accessing to complete output logits (\ie, \textit{logits-based attacks}), which are usually not available in practice. In this paper, we study the vulnerability of pre-trained LLMs to MIAs in the \textit{label-only setting}, where the adversary can only access generated tokens (text). We first reveal that existing label-only MIAs have minor effects in attacking pre-trained LLMs, although they are highly effective in inferring fine-tuning datasets used for personalized LLMs. We find that their failure stems from two main reasons, including better generalization and overly coarse perturbation. Specifically, due to the extensive pre-training corpora and exposing each sample only a few times, LLMs exhibit minimal robustness differences between members and non-members. This makes token-level perturbations too coarse to capture such differences.   To alleviate these problems, we propose \textbf{PETAL}: a label-only membership inference attack based on \textbf{PE}r-\textbf{T}oken sem\textbf{A}ntic simi\textbf{L}arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are `better' memorized and have smaller perplexity. We conduct extensive experiments on the WikiMIA benchmark and the more challenging MIMIR benchmark. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics on five prevalent open-source LLMs.

摘要: 成员推理攻击(MIA)的目的是预测数据样本是否属于模型的训练集。尽管先前的研究已经广泛地探索了大型语言模型(LLM)中的MIA，但它们通常需要访问以完成输出日志(即，文本{基于日志的攻击})，而这在实践中通常是不可用的。在该文中，我们研究了在仅标签设置的情况下，攻击者只能访问生成的令牌(文本)的情况下，预先训练的LLMS对MIA的脆弱性。我们首先揭示了现有的仅标签MIA在攻击预先训练的LLM方面的影响很小，尽管它们在推断用于个性化LLM的微调数据集方面非常有效。我们发现，它们的失败有两个主要原因，包括较好的泛化和过粗的扰动。具体地说，由于大量的预训练语料库和每个样本只暴露几次，LLMS在成员和非成员之间表现出最小的稳健性差异。这使得令牌级的扰动过于粗略，无法捕捉到这样的差异。为了缓解这些问题，我们提出了一种基于Textbf{PE}r-\Textbf{T}Oken Sem\Textbf{A}Ntic Simi\Textbf{L}的仅标签成员关系推理攻击。具体地说，Petal利用令牌级语义相似性来近似输出概率，并随后计算困惑。它最终基于一个共同的假设来揭示成员身份，即成员的记忆力更好，困惑程度更小。我们在WikiMIA基准和更具挑战性的Mimir基准上进行了广泛的实验。根据经验，我们的Petal在针对个性化LLM的现有仅标签攻击的扩展上性能更好，甚至在五个流行的开源LLM上的所有指标上都与其他基于Logit的高级攻击相当。



## **32. Adversarial Universal Stickers: Universal Perturbation Attacks on Traffic Sign using Stickers**

对抗通用贴纸：使用贴纸对交通标志进行通用扰动攻击 cs.CV

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18724v1) [paper-pdf](http://arxiv.org/pdf/2502.18724v1)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial attacks on deep learning models have proliferated in recent years. In many cases, a different adversarial perturbation is required to be added to each image to cause the deep learning model to misclassify it. This is ineffective as each image has to be modified in a different way. Meanwhile, research on universal perturbations focuses on designing a single perturbation that can be applied to all images in a data set, and cause a deep learning model to misclassify the images. This work advances the field of universal perturbations by exploring universal perturbations in the context of traffic signs and autonomous vehicle systems. This work introduces a novel method for generating universal perturbations that visually look like simple black and white stickers, and using them to cause incorrect street sign predictions. Unlike traditional adversarial perturbations, the adversarial universal stickers are designed to be applicable to any street sign: same sticker, or stickers, can be applied in same location to any street sign and cause it to be misclassified. Further, to enable safe experimentation with adversarial images and street signs, this work presents a virtual setting that leverages Street View images of street signs, rather than the need to physically modify street signs, to test the attacks. The experiments in the virtual setting demonstrate that these stickers can consistently mislead deep learning models used commonly in street sign recognition, and achieve high attack success rates on dataset of US traffic signs. The findings highlight the practical security risks posed by simple stickers applied to traffic signs, and the ease with which adversaries can generate adversarial universal stickers that can be applied to many street signs.

摘要: 近年来，对深度学习模型的敌意攻击激增。在许多情况下，需要向每个图像添加不同的对抗性扰动，以使深度学习模型对其进行错误分类。这是无效的，因为每个图像都必须以不同的方式进行修改。同时，对普遍扰动的研究集中在设计一个单一的扰动，该单一扰动可以应用于数据集中的所有图像，并导致深度学习模型对图像进行错误分类。这项工作通过探索交通标志和自动车辆系统中的普遍扰动，推动了普遍扰动领域的发展。这项工作介绍了一种新的方法来产生普遍的扰动，视觉上看起来像简单的黑白贴纸，并使用它们来导致错误的街道标志预测。与传统的对抗性干扰不同，对抗性通用贴纸被设计为适用于任何路牌：相同的贴纸，或多个贴纸，可以在相同的位置贴到任何路标上，并导致分类错误。此外，为了实现对抗性图像和街道标志的安全试验，这项工作提供了一个虚拟环境，该环境利用街道标志的街景图像来测试攻击，而不需要物理修改街道标志。在虚拟环境中的实验表明，这些贴纸能够一致地误导路牌识别中常用的深度学习模型，并在美国交通标志数据集上取得了较高的攻击成功率。这些发现突显了应用于交通标志的简单贴纸带来的实际安全风险，以及对手可以很容易地生成可以应用于许多街道标志的对抗性通用贴纸。



## **33. Time Traveling to Defend Against Adversarial Example Attacks in Image Classification**

时间旅行以防御对抗图像分类中的示例攻击 cs.CR

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2410.08338v2) [paper-pdf](http://arxiv.org/pdf/2410.08338v2)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial example attacks have emerged as a critical threat to machine learning. Adversarial attacks in image classification abuse various, minor modifications to the image that confuse the image classification neural network -- while the image still remains recognizable to humans. One important domain where the attacks have been applied is in the automotive setting with traffic sign classification. Researchers have demonstrated that adding stickers, shining light, or adding shadows are all different means to make machine learning inference algorithms mis-classify the traffic signs. This can cause potentially dangerous situations as a stop sign is recognized as a speed limit sign causing vehicles to ignore it and potentially leading to accidents. To address these attacks, this work focuses on enhancing defenses against such adversarial attacks. This work shifts the advantage to the user by introducing the idea of leveraging historical images and majority voting. While the attacker modifies a traffic sign that is currently being processed by the victim's machine learning inference, the victim can gain advantage by examining past images of the same traffic sign. This work introduces the notion of ''time traveling'' and uses historical Street View images accessible to anybody to perform inference on different, past versions of the same traffic sign. In the evaluation, the proposed defense has 100% effectiveness against latest adversarial example attack on traffic sign classification algorithm.

摘要: 对抗性例子攻击已经成为机器学习的一个严重威胁。图像分类中的对抗性攻击利用了对图像的各种微小修改，这混淆了图像分类神经网络--同时图像仍然可以被人类识别。应用攻击的一个重要领域是具有交通标志分类的汽车环境。研究人员已经证明，添加贴纸、照亮灯光或添加阴影都是使机器学习推理算法错误分类交通标志的不同方法。这可能会导致潜在的危险情况，因为停车标志被识别为限速标志，导致车辆忽略它，并可能导致事故。为了应对这些攻击，这项工作的重点是加强对这种对抗性攻击的防御。这项工作通过引入利用历史图像和多数投票的想法将优势转移到用户身上。当攻击者修改当前正在由受害者的机器学习推理处理的交通标志时，受害者可以通过检查同一交通标志的过去图像来获得优势。这项工作引入了时间旅行的概念，并使用任何人都可以访问的历史街景图像来对同一交通标志的不同过去版本进行推断。在评估中，所提出的防御措施对交通标志分类算法的最新对手例攻击具有100%的有效性。



## **34. Fall Leaf Adversarial Attack on Traffic Sign Classification**

交通标志分类的落叶对抗攻击 cs.CV

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2411.18776v2) [paper-pdf](http://arxiv.org/pdf/2411.18776v2)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial input image perturbation attacks have emerged as a significant threat to machine learning algorithms, particularly in image classification setting. These attacks involve subtle perturbations to input images that cause neural networks to misclassify the input images, even though the images remain easily recognizable to humans. One critical area where adversarial attacks have been demonstrated is in automotive systems where traffic sign classification and recognition is critical, and where misclassified images can cause autonomous systems to take wrong actions. This work presents a new class of adversarial attacks. Unlike existing work that has focused on adversarial perturbations that leverage human-made artifacts to cause the perturbations, such as adding stickers, paint, or shining flashlights at traffic signs, this work leverages nature-made artifacts: tree leaves. By leveraging nature-made artifacts, the new class of attacks has plausible deniability: a fall leaf stuck to a street sign could come from a near-by tree, rather than be placed there by an malicious human attacker. To evaluate the new class of the adversarial input image perturbation attacks, this work analyses how fall leaves can cause misclassification in street signs. The work evaluates various leaves from different species of trees, and considers various parameters such as size, color due to tree leaf type, and rotation. The work demonstrates high success rate for misclassification. The work also explores the correlation between successful attacks and how they affect the edge detection, which is critical in many image classification algorithms.

摘要: 对抗性输入图像扰动攻击已经成为机器学习算法的一个重大威胁，特别是在图像分类设置中。这些攻击涉及对输入图像的微妙扰动，导致神经网络对输入图像进行错误分类，即使图像仍然很容易被人类识别。已演示对抗性攻击的一个关键领域是在交通标志分类和识别至关重要的汽车系统中，错误分类的图像可能会导致自动系统采取错误的操作。这项工作提出了一类新的对抗性攻击。与现有的专注于对抗性干扰的工作不同，这些工作利用人造人工制品来引起扰动，例如添加贴纸、油漆或向交通标志照射手电筒，而这项工作利用了自然制造的人工制品：树叶。通过利用自然制造的文物，这种新的攻击类别具有看似合理的抵赖性：粘在路牌上的落叶可能来自附近的树，而不是被恶意的人类攻击者放置在那里。为了评估这类新的对抗性输入图像扰动攻击，本工作分析了落叶如何导致街道标志的误分类。这项工作评估了不同树种的各种树叶，并考虑了各种参数，如大小、树叶类型造成的颜色和旋转。这项工作表明，错误分类的成功率很高。这项工作还探索了成功的攻击之间的相关性以及它们如何影响边缘检测，这在许多图像分类算法中是至关重要的。



## **35. Toward Breaking Watermarks in Distortion-free Large Language Models**

迈向无失真大型语言模型中的水印 cs.CR

5 pages, AAAI'25 Workshop on Preventing and Detecting LLM Generated  Misinformation

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18608v1) [paper-pdf](http://arxiv.org/pdf/2502.18608v1)

**Authors**: Shayleen Reynolds, Saheed Obitayo, Niccolò Dalmasso, Dung Daniel T. Ngo, Vamsi K. Potluru, Manuela Veloso

**Abstract**: In recent years, LLM watermarking has emerged as an attractive safeguard against AI-generated content, with promising applications in many real-world domains. However, there are growing concerns that the current LLM watermarking schemes are vulnerable to expert adversaries wishing to reverse-engineer the watermarking mechanisms. Prior work in "breaking" or "stealing" LLM watermarks mainly focuses on the distribution-modifying algorithm of Kirchenbauer et al. (2023), which perturbs the logit vector before sampling. In this work, we focus on reverse-engineering the other prominent LLM watermarking scheme, distortion-free watermarking (Kuditipudi et al. 2024), which preserves the underlying token distribution by using a hidden watermarking key sequence. We demonstrate that, even under a more sophisticated watermarking scheme, it is possible to "compromise" the LLM and carry out a "spoofing" attack. Specifically, we propose a mixed integer linear programming framework that accurately estimates the secret key used for watermarking using only a few samples of the watermarked dataset. Our initial findings challenge the current theoretical claims on the robustness and usability of existing LLM watermarking techniques.

摘要: 近年来，LLM水印作为一种针对人工智能生成的内容的有吸引力的保护措施，在许多现实世界领域都有很好的应用前景。然而，越来越多的人担心当前的LLM水印方案容易受到希望对水印机制进行反向工程的专家对手的攻击。以往对LLM水印“破解”或“窃取”的研究主要集中在Kirchenbauer等人的分布修正算法上。(2023)，在采样之前对Logit向量进行扰动。在这项工作中，我们专注于对另一种著名的LLM水印方案--无失真水印(Kuditipudi等人)进行逆向工程。2024)，其通过使用隐藏的水印密钥序列来保留底层令牌分布。我们证明，即使在更复杂的水印方案下，也有可能“危害”LLM并执行“欺骗”攻击。具体地说，我们提出了一种混合整数线性规划框架，该框架仅使用几个水印数据集的样本就可以准确地估计用于水印的秘密密钥。我们的初步发现挑战了当前关于现有LLM水印技术的稳健性和可用性的理论主张。



## **36. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.01386v2) [paper-pdf](http://arxiv.org/pdf/2502.01386v2)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型(LLMS)的检索-增强生成(RAG)系统已经成为诸如问题回答和内容生成等任务的关键。然而，由于其固有的脆弱性，它们对舆论和信息传播的影响越来越大，使其成为安全研究的关键焦点。以前的研究主要针对针对事实或单一查询操作的攻击。在本文中，我们讨论了一个更实际的场景：针对RAG模型的面向主题的对抗性意见操纵攻击，其中要求LLM推理和综合多个视角，使它们特别容易受到系统性知识中毒的影响。具体地说，我们提出了Theme-FlipRAG，这是一种两阶段操纵攻击管道，战略性地制造对抗性扰动来影响相关查询的观点。该方法结合了传统的对抗性排序攻击技术，并利用LLMS丰富的内部相关知识和推理能力来执行语义级的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显著影响了用户对信息的感知。目前的缓解方法无法有效防御此类攻击，这突显了加强RAG系统安全保障的必要性，并为LLM安全研究提供了至关重要的见解。



## **37. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

CLIPure：通过CLIP在潜空间中净化，以实现对抗鲁棒零镜头分类 cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18176v1) [paper-pdf](http://arxiv.org/pdf/2502.18176v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.

摘要: 在这篇文章中，我们的目标是建立一个对抗性稳健的零镜头图像分类器。我们的工作基于CLIP，这是一个视觉语言预先训练的编码器模型，它可以通过将图像与文本提示进行匹配来执行零镜头分类。净化是我们选择的路径，因为它不需要针对特定攻击类型的对抗性训练，因此可以应对任何可预见的攻击。然后，我们通过双向随机微分方程(SDE)将净化风险表示为对敌方样本去噪的净化过程和对良性样本添加扰动的攻击过程的联合分布之间的KL发散。最终得出的结果启发我们去探索CLIP的多峰潜伏空间中的净化。我们为我们的CLIPure方法提出了两种变体：CLIPure-Diff和CLIPure-Cos，CLIPure-Diff使用DALE-2中的DiffusionPrior模块(对剪辑的潜在向量的生成过程进行建模)来模拟图像的潜在向量的可能性，CLIPure-Cos使用图像的嵌入和“a的照片”之间的余弦相似性来建模可能性。据我们所知，CLIPure是第一个在多峰潜在空间中进行净化的方法，而CLIPure-Cos是第一个不基于产生式模型的净化方法，大大提高了防御效率。我们在CIFAR-10、ImageNet和13个数据集上进行了广泛的实验，这些数据集是以前基于剪辑的防御方法用于评估零镜头分类稳健性的。结果表明，CLIPure在很大程度上提高了SOTA的健壮性，例如，在CIFAR10上从71.7%提高到91.1%，在ImageNet上从59.6%提高到72.6%，在13个数据集上的平均健壮性比以前的SOTA提高了108%。代码可在https://github.com/TMLResearchGroup-CAS/CLIPure.上获得



## **38. Exploring the Robustness and Transferability of Patch-Based Adversarial Attacks in Quantized Neural Networks**

探索量化神经网络中基于补丁的对抗攻击的鲁棒性和可移植性 cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.15246v2) [paper-pdf](http://arxiv.org/pdf/2411.15246v2)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized neural networks (QNNs) are increasingly used for efficient deployment of deep learning models on resource-constrained platforms, such as mobile devices and edge computing systems. While quantization reduces model size and computational demands, its impact on adversarial robustness-especially against patch-based attacks-remains inadequately addressed. Patch-based attacks, characterized by localized, high-visibility perturbations, pose significant security risks due to their transferability and resilience. In this study, we systematically evaluate the vulnerability of QNNs to patch-based adversarial attacks across various quantization levels and architectures, focusing on factors that contribute to the robustness of these attacks. Through experiments analyzing feature representations, quantization strength, gradient alignment, and spatial sensitivity, we find that patch attacks consistently achieve high success rates across bitwidths and architectures, demonstrating significant transferability even in heavily quantized models. Contrary to the expectation that quantization might enhance adversarial defenses, our results show that QNNs remain highly susceptible to patch attacks due to the persistence of distinct, localized features within quantized representations. These findings underscore the need for quantization-aware defenses that address the specific challenges posed by patch-based attacks. Our work contributes to a deeper understanding of adversarial robustness in QNNs and aims to guide future research in developing secure, quantization-compatible defenses for real-world applications.

摘要: 量化神经网络(QNN)越来越多地被用于在资源受限的平台上高效地部署深度学习模型，例如移动设备和边缘计算系统。虽然量化减少了模型大小和计算需求，但它对对手健壮性的影响--特别是针对基于补丁的攻击--仍然没有得到充分的解决。基于补丁的攻击，其特点是局部化、高可见性的扰动，由于其可转移性和弹性，构成了巨大的安全风险。在这项研究中，我们系统地评估了QNN在不同量化级别和体系结构上对基于补丁的攻击的脆弱性，重点讨论了影响这些攻击的健壮性的因素。通过对特征表示、量化强度、梯度对齐和空间敏感度的实验分析，我们发现补丁攻击在不同的比特和体系结构上都获得了很高的成功率，即使在高度量化的模型中也表现出了显著的可移植性。与量化可能增强敌意防御的预期相反，我们的结果表明，由于量化表示中独特的局部化特征的持久性，QNN仍然非常容易受到补丁攻击。这些发现强调了量化感知防御的必要性，以应对基于补丁的攻击带来的具体挑战。我们的工作有助于更深入地理解QNN中的对抗健壮性，并旨在指导未来的研究，为现实世界的应用开发安全的、量化兼容的防御措施。



## **39. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

迈向强大和安全的人工智能：关于漏洞和攻击的调查 cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13175v2) [paper-pdf](http://arxiv.org/pdf/2502.13175v2)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.

摘要: 包括机器人和自动驾驶车辆在内的具体化人工智能系统正越来越多地融入现实世界的应用程序，在这些应用程序中，它们遇到了一系列源于环境和系统层面因素的漏洞。这些漏洞表现为传感器欺骗、对抗性攻击以及任务和运动规划中的失败，对健壮性和安全性构成了重大挑战。尽管研究的主体越来越多，但现有的审查很少专门关注嵌入式人工智能系统的独特安全和安保挑战。大多数以前的工作要么解决了一般的人工智能漏洞，要么专注于孤立的方面，缺乏一个专门为体现的人工智能量身定做的统一框架。本调查通过以下方式填补这一关键空白：(1)将特定于具身人工智能的漏洞分为外源性(如物理攻击、网络安全威胁)和内源性(如传感器故障、软件缺陷)来源；(2)系统分析具身人工智能特有的对抗性攻击范式，重点关注它们对感知、决策和具身交互的影响；(3)调查针对具身系统内的大视觉语言模型(LVLM)和大语言模型(LMS)的攻击向量，如越狱攻击和指令曲解；(4)评估体现感知、决策和任务规划算法中的健壮性挑战；(5)提出有针对性的策略，以提高体现人工智能系统的安全性和可靠性。通过整合这些维度，我们提供了一个全面的框架，用于理解体现的人工智能中漏洞和安全之间的相互作用。



## **40. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2410.23091v7) [paper-pdf](http://arxiv.org/pdf/2410.23091v7)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.上获得



## **41. Towards Certification of Uncertainty Calibration under Adversarial Attacks**

对抗攻击下的不确定性校准认证 cs.LG

10 pages main paper, appendix included Published at: International  Conference on Learning Representations (ICLR) 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2405.13922v3) [paper-pdf](http://arxiv.org/pdf/2405.13922v3)

**Authors**: Cornelius Emde, Francesco Pinto, Thomas Lukasiewicz, Philip H. S. Torr, Adel Bibi

**Abstract**: Since neural classifiers are known to be sensitive to adversarial perturbations that alter their accuracy, \textit{certification methods} have been developed to provide provable guarantees on the insensitivity of their predictions to such perturbations. Furthermore, in safety-critical applications, the frequentist interpretation of the confidence of a classifier (also known as model calibration) can be of utmost importance. This property can be measured via the Brier score or the expected calibration error. We show that attacks can significantly harm calibration, and thus propose certified calibration as worst-case bounds on calibration under adversarial perturbations. Specifically, we produce analytic bounds for the Brier score and approximate bounds via the solution of a mixed-integer program on the expected calibration error. Finally, we propose novel calibration attacks and demonstrate how they can improve model calibration through \textit{adversarial calibration training}.

摘要: 由于众所周知，神经分类器对改变其准确性的对抗性扰动敏感，因此\textit{认证方法}的开发是为了提供可证明的保证其预测对此类扰动的不敏感性。此外，在安全关键应用中，分类器置信度的频率主义解释（也称为模型校准）可能至关重要。该属性可以通过Brier评分或预期的校准误差来测量。我们表明，攻击可能会严重损害校准，因此建议将经过认证的校准作为对抗性扰动下校准的最坏情况界限。具体来说，我们通过对预期校准误差求解混合整数程序来产生Brier分数的分析界限和近似界限。最后，我们提出了新颖的校准攻击，并演示了它们如何通过\textit{对抗校准训练}来改进模型校准。



## **42. Model-Free Adversarial Purification via Coarse-To-Fine Tensor Network Representation**

通过粗到细张量网络表示的无模型对抗净化 cs.LG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17972v1) [paper-pdf](http://arxiv.org/pdf/2502.17972v1)

**Authors**: Guang Lin, Duc Thien Nguyen, Zerui Tao, Konstantinos Slavakis, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Deep neural networks are known to be vulnerable to well-designed adversarial attacks. Although numerous defense strategies have been proposed, many are tailored to the specific attacks or tasks and often fail to generalize across diverse scenarios. In this paper, we propose Tensor Network Purification (TNP), a novel model-free adversarial purification method by a specially designed tensor network decomposition algorithm. TNP depends neither on the pre-trained generative model nor the specific dataset, resulting in strong robustness across diverse adversarial scenarios. To this end, the key challenge lies in relaxing Gaussian-noise assumptions of classical decompositions and accommodating the unknown distribution of adversarial perturbations. Unlike the low-rank representation of classical decompositions, TNP aims to reconstruct the unobserved clean examples from an adversarial example. Specifically, TNP leverages progressive downsampling and introduces a novel adversarial optimization objective to address the challenge of minimizing reconstruction error but without inadvertently restoring adversarial perturbations. Extensive experiments conducted on CIFAR-10, CIFAR-100, and ImageNet demonstrate that our method generalizes effectively across various norm threats, attack types, and tasks, providing a versatile and promising adversarial purification technique.

摘要: 众所周知，深度神经网络很容易受到精心设计的对抗性攻击。尽管已经提出了许多防御策略，但许多都是针对特定的攻击或任务量身定做的，往往无法对不同的场景进行概括。在本文中，我们提出了张量网络净化(TNP)，这是一种新的无模型的对抗性净化方法，它通过专门设计的张量网络分解算法来实现。TNP既不依赖于预先训练的生成模型，也不依赖于特定的数据集，从而在不同的对抗场景中具有很强的稳健性。为此，关键的挑战在于放宽经典分解的高斯噪声假设，并适应对抗性扰动的未知分布。与经典分解的低阶表示不同，TNP的目标是从对抗性实例中重构未被观察到的干净实例。具体地说，TNP利用渐进式下采样，并引入了一种新的对抗性优化目标来解决最小化重建误差但不会无意中恢复对抗性扰动的挑战。在CIFAR-10、CIFAR-100和ImageNet上进行的大量实验表明，我们的方法有效地概括了各种规范威胁、攻击类型和任务，提供了一种通用的、有前景的对手净化技术。



## **43. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

LiSA：利用链接推荐通过子图注入攻击图神经网络 cs.LG

PAKDD 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.09271v3) [paper-pdf](http://arxiv.org/pdf/2502.09271v3)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.

摘要: 图神经网络(GNN)在用图结构建模数据方面表现出了卓越的能力，但最近的研究表明，它们对对手攻击很敏感。传统的攻击方法依赖于操纵原始图形或添加到人工创建的节点的链接，在现实世界中往往被证明是不切实际的。在GNN系统中，引入一个孤立的子图来欺骗链接推荐器和节点分类器，提出了一种新的对抗性场景。具体地说，链接推荐器被误导提出目标受害节点与子图之间的链接，鼓励用户无意中建立连接，这将降低节点分类的准确性，从而促进攻击的成功。为了解决这一问题，我们提出了LISA框架，该框架采用双重代理模型和双层优化来同时满足两个对抗性目标。在真实数据集上的大量实验证明了该方法的有效性。



## **44. Relationship between Uncertainty in DNNs and Adversarial Attacks**

DNN的不确定性与对抗攻击之间的关系 cs.LG

review

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2409.13232v2) [paper-pdf](http://arxiv.org/pdf/2409.13232v2)

**Authors**: Mabel Ogonna, Abigail Adeniran, Adewale Adeyemo

**Abstract**: Deep Neural Networks (DNNs) have achieved state of the art results and even outperformed human accuracy in many challenging tasks, leading to DNNs adoption in a variety of fields including natural language processing, pattern recognition, prediction, and control optimization. However, DNNs are accompanied by uncertainty about their results, causing them to predict an outcome that is either incorrect or outside of a certain level of confidence. These uncertainties stem from model or data constraints, which could be exacerbated by adversarial attacks. Adversarial attacks aim to provide perturbed input to DNNs, causing the DNN to make incorrect predictions or increase model uncertainty. In this review, we explore the relationship between DNN uncertainty and adversarial attacks, emphasizing how adversarial attacks might raise DNN uncertainty.

摘要: 深度神经网络（DNN）已实现最先进的结果，甚至在许多具有挑战性的任务中超过了人类的准确性，导致DNN在自然语言处理、模式识别、预测和控制优化等各个领域得到采用。然而，DNN伴随着结果的不确定性，导致它们预测的结果要么不正确，要么超出一定置信水平。这些不确定性源于模型或数据限制，对抗性攻击可能会加剧这种限制。对抗性攻击旨在向DNN提供受干扰的输入，导致DNN做出错误的预测或增加模型的不确定性。在这篇评论中，我们探讨了DNN不确定性和对抗性攻击之间的关系，强调了对抗性攻击如何提高DNN不确定性。



## **45. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**

大型语言模型中拒绝的几何学：概念锥和表示独立性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17420v1) [paper-pdf](http://arxiv.org/pdf/2502.17420v1)

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.

摘要: 大型语言模型(LLM)的安全一致性可以通过恶意创建的输入来规避，但这些攻击绕过安全屏障的机制仍然知之甚少。先前的工作表明，模型激活空间中的单个拒绝方向决定了LLM是否拒绝请求。在这项研究中，我们提出了一种新的基于梯度的表示工程方法，并用它来识别拒绝方向。与以前的工作相反，我们发现了多个独立的方向，甚至是调解拒绝的多维概念锥。此外，我们表明，正交性本身并不意味着干预下的独立性，这激发了既能解释线性效应又能解释非线性效应的表征独立性的概念。利用这个框架，我们确定了机械独立的拒绝方向。我们发现，LLMS中的拒绝机制受到复杂空间结构的支配，并识别出功能独立的方向，证实了多种不同的机制驱动着拒绝行为。我们的基于梯度的方法揭示了这些机制，并可以进一步作为理解LLMS的未来工作的基础。



## **46. Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences**

伪攻击：通过伪君子序列对NLP系统进行零微扰对抗攻击 cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17392v1) [paper-pdf](http://arxiv.org/pdf/2502.17392v1)

**Authors**: Yangshijie Zhang

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.   In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems.

摘要: 深度神经网络(DNN)在自然语言处理(NLP)领域取得了显著的成功，产生了广泛认可的应用，如ChatGPT。然而，这些模型对对抗性攻击的脆弱性仍然是一个重大关切。与图像等连续领域不同，文本存在于离散的空间中，即使是句子、单词或字符级别的微小更改也很容易被人类察觉。这种固有的离散性也使传统优化技术的使用变得复杂，因为文本是不可区分的。以往对文本中敌意攻击的研究主要集中在字符级、词级、句子级和多层方法上，所有这些方法都存在效率低下或可感知性问题，原因是需要进行多个查询或显著的语义转换。在这项工作中，我们介绍了一种新的对抗性攻击方法，Emoji-Attack，它利用表情符号的操纵来制造微妙但有效的扰动。与字符和单词级别的策略不同，Emoji攻击将表情符号作为不同的攻击层，导致不太明显的变化，对文本的破坏最小。这种方法在以前的研究中基本上没有被探索过，这些研究通常专注于将表情符号插入作为字符级攻击的扩展。我们的实验表明，Emoji-Attack在大小模型上都具有很强的攻击性能，是一种在NLP系统中增强对手健壮性的很有前途的技术。



## **47. On the Vulnerability of Concept Erasure in Diffusion Models**

扩散模型中概念擦除的脆弱性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17537v1) [paper-pdf](http://arxiv.org/pdf/2502.17537v1)

**Authors**: Lucas Beerens, Alex D. Richardson, Kaicheng Zhang, Dongdong Chen

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. To address these issues, research on machine unlearning has developed various concept erasure methods, which aim to remove the effect of unwanted data through post-hoc training. However, we show these erasure techniques are vulnerable, where images of supposedly erased concepts can still be generated using adversarially crafted prompts. We introduce RECORD, a coordinate-descent-based algorithm that discovers prompts capable of eliciting the generation of erased content. We demonstrate that RECORD significantly beats the attack success rate of current state-of-the-art attack methods. Furthermore, our findings reveal that models subjected to concept erasure are more susceptible to adversarial attacks than previously anticipated, highlighting the urgency for more robust unlearning approaches. We open source all our code at https://github.com/LucasBeerens/RECORD

摘要: 文本到图像传播模式的激增引起了对隐私和安全的严重关切，特别是关于产生受版权保护或有害的图像。为了解决这些问题，机器遗忘的研究已经发展出各种概念擦除方法，旨在通过后自组织训练来消除不需要的数据的影响。然而，我们发现这些擦除技术是脆弱的，在这些技术中，应该被擦除的概念的图像仍然可以使用相反的精心制作的提示来生成。我们介绍了Record，这是一种基于坐标下降的算法，它发现能够引发删除内容生成的提示。我们证明，RECORD大大超过了当前最先进的攻击方法的攻击成功率。此外，我们的发现显示，受到概念删除的模型比之前预期的更容易受到对抗性攻击，这突显了更强大的遗忘方法的紧迫性。我们在https://github.com/LucasBeerens/RECORD上开放了我们所有的代码



## **48. Order Fairness Evaluation of DAG-based ledgers**

基于DAB的分类帐的订单公平性评估 cs.CR

19 pages with 9 pages dedicated to references and appendices, 23  figures, 13 of which are in the appendices

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17270v1) [paper-pdf](http://arxiv.org/pdf/2502.17270v1)

**Authors**: Erwan Mahe, Sara Tucci-Piergiovanni

**Abstract**: Order fairness in distributed ledgers refers to properties that relate the order in which transactions are sent or received to the order in which they are eventually finalized, i.e., totally ordered. The study of such properties is relatively new and has been especially stimulated by the rise of Maximal Extractable Value (MEV) attacks in blockchain environments. Indeed, in many classical blockchain protocols, leaders are responsible for selecting the transactions to be included in blocks, which creates a clear vulnerability and opportunity for transaction order manipulation.   Unlike blockchains, DAG-based ledgers allow participants in the network to independently propose blocks, which are then arranged as vertices of a directed acyclic graph. Interestingly, leaders in DAG-based ledgers are elected only after the fact, once transactions are already part of the graph, to determine their total order. In other words, transactions are not chosen by single leaders; instead, they are collectively validated by the nodes, and leaders are only elected to establish an ordering. This approach intuitively reduces the risk of transaction manipulation and enhances fairness.   In this paper, we aim to quantify the capability of DAG-based ledgers to achieve order fairness. To this end, we define new variants of order fairness adapted to DAG-based ledgers and evaluate the impact of an adversary capable of compromising a limited number of nodes (below the one-third threshold) to reorder transactions. We analyze how often our order fairness properties are violated under different network conditions and parameterizations of the DAG algorithm, depending on the adversary's power.   Our study shows that DAG-based ledgers are still vulnerable to reordering attacks, as an adversary can coordinate a minority of Byzantine nodes to manipulate the DAG's structure.

摘要: 分布式分类账中的顺序公平性是指将发送或接收交易的顺序与最终确定的顺序(即完全有序)联系起来的属性。对这类属性的研究相对较新，尤其是区块链环境中最大可提取价值(MEV)攻击的兴起。事实上，在许多经典的区块链协议中，领导者负责选择要包含在区块中的交易，这为交易顺序操纵创造了明显的漏洞和机会。与区块链不同，基于DAG的分类账允许网络中的参与者独立提出区块，然后将这些区块排列为有向无环图的顶点。有趣的是，只有在交易已经成为图表的一部分后，才会在基于DAG的分类账中选出领导人，以确定其总顺序。换句话说，事务不是由单个领导者选择的；相反，它们由节点集体验证，而领导者只被选举来建立顺序。这种方法直观地降低了交易操纵的风险，提高了公平性。在本文中，我们的目标是量化基于DAG的分类帐实现顺序公平的能力。为此，我们定义了适用于基于DAG的分类账的顺序公平性的新变体，并评估了攻击者能够危害有限数量的节点(低于三分之一的阈值)来重新排序事务的影响。我们分析了在不同的网络条件和DAG算法的参数设置下，我们的顺序公平性被违反的频率，这取决于对手的力量。我们的研究表明，基于DAG的分类账仍然容易受到重新排序攻击，因为对手可以协调少数拜占庭节点来操纵DAG的结构。



## **49. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective**

REINFORCE对大型语言模型的对抗攻击：自适应、分布和语义目标 cs.LG

30 pages, 6 figures, 15 tables

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17254v1) [paper-pdf](http://arxiv.org/pdf/2502.17254v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann

**Abstract**: To circumvent the alignment of large language models (LLMs), current optimization-based adversarial attacks usually craft adversarial prompts by maximizing the likelihood of a so-called affirmative response. An affirmative response is a manually designed start of a harmful answer to an inappropriate request. While it is often easy to craft prompts that yield a substantial likelihood for the affirmative response, the attacked model frequently does not complete the response in a harmful manner. Moreover, the affirmative objective is usually not adapted to model-specific preferences and essentially ignores the fact that LLMs output a distribution over responses. If low attack success under such an objective is taken as a measure of robustness, the true robustness might be grossly overestimated. To alleviate these flaws, we propose an adaptive and semantic optimization problem over the population of responses. We derive a generally applicable objective via the REINFORCE policy-gradient formalism and demonstrate its efficacy with the state-of-the-art jailbreak algorithms Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD). For example, our objective doubles the attack success rate (ASR) on Llama3 and increases the ASR from 2% to 50% with circuit breaker defense.

摘要: 为了绕过大型语言模型(LLM)的对齐，当前基于优化的对抗性攻击通常通过最大化所谓肯定响应的可能性来创建对抗性提示。肯定答复是手动设计的对不适当请求的有害答复的开始。虽然通常很容易制定提示，以产生肯定响应的很大可能性，但被攻击的模型通常不会以有害的方式完成响应。此外，肯定的目标通常不适应特定于模型的偏好，基本上忽略了LLMS输出的分布高于响应的事实。如果在这样的目标下将低攻击成功率作为稳健性的衡量标准，则可能严重高估了真正的稳健性。为了克服这些缺陷，我们提出了一种基于响应总体的自适应语义优化问题。我们通过强化策略梯度理论推导出一个普遍适用的目标，并用最先进的越狱算法贪婪坐标梯度(GCG)和投影梯度下降(PGD)来验证其有效性。例如，我们的目标是使Llama3上的攻击成功率(ASR)翻一番，并通过断路器防御将ASR从2%提高到50%。



## **50. Adversarial Training for Defense Against Label Poisoning Attacks**

防御标签中毒攻击的对抗训练 cs.LG

Accepted at the International Conference on Learning Representations  (ICLR 2025)

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17121v1) [paper-pdf](http://arxiv.org/pdf/2502.17121v1)

**Authors**: Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach

**Abstract**: As machine learning models grow in complexity and increasingly rely on publicly sourced data, such as the human-annotated labels used in training large language models, they become more vulnerable to label poisoning attacks. These attacks, in which adversaries subtly alter the labels within a training dataset, can severely degrade model performance, posing significant risks in critical applications. In this paper, we propose FLORAL, a novel adversarial training defense strategy based on support vector machines (SVMs) to counter these threats. Utilizing a bilevel optimization framework, we cast the training process as a non-zero-sum Stackelberg game between an attacker, who strategically poisons critical training labels, and the model, which seeks to recover from such attacks. Our approach accommodates various model architectures and employs a projected gradient descent algorithm with kernel SVMs for adversarial training. We provide a theoretical analysis of our algorithm's convergence properties and empirically evaluate FLORAL's effectiveness across diverse classification tasks. Compared to robust baselines and foundation models such as RoBERTa, FLORAL consistently achieves higher robust accuracy under increasing attacker budgets. These results underscore the potential of FLORAL to enhance the resilience of machine learning models against label poisoning threats, thereby ensuring robust classification in adversarial settings.

摘要: 随着机器学习模型变得越来越复杂，并越来越依赖于公共来源的数据，例如用于训练大型语言模型的人类注释标签，它们变得更容易受到标签中毒攻击。在这些攻击中，攻击者巧妙地更改了训练数据集中的标签，可能会严重降低模型的性能，给关键应用程序带来重大风险。针对这些威胁，本文提出了一种新的基于支持向量机的对抗性训练防御策略FLOLAR。利用双层优化框架，我们将训练过程描述为攻击者和模型之间的非零和Stackelberg博弈，攻击者策略性地毒害关键的训练标签，而模型试图从此类攻击中恢复。我们的方法适应了不同的模型结构，并使用了一种带有核支持向量机的投影梯度下降算法进行对抗性训练。我们对算法的收敛特性进行了理论分析，并对FLORAL算法在不同分类任务上的有效性进行了实证评估。与罗伯塔等稳健的基线和基础模型相比，FLORAL在不断增加的攻击者预算下始终实现更高的稳健精度。这些结果强调了FLOLAR的潜力，以增强机器学习模型对标签中毒威胁的弹性，从而确保在对抗性环境中的稳健分类。



