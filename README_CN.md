# Latest Adversarial Attack Papers
**update at 2025-07-03 09:20:22**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Boosting Adversarial Transferability Against Defenses via Multi-Scale Transformation**

通过多规模转型提高针对防守的对抗转移能力 cs.CV

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01791v1) [paper-pdf](http://arxiv.org/pdf/2507.01791v1)

**Authors**: Zihong Guo, Chen Wan, Yayin Zheng, Hailing Kuang, Xiaohai Lu

**Abstract**: The transferability of adversarial examples poses a significant security challenge for deep neural networks, which can be attacked without knowing anything about them. In this paper, we propose a new Segmented Gaussian Pyramid (SGP) attack method to enhance the transferability, particularly against defense models. Unlike existing methods that generally focus on single-scale images, our approach employs Gaussian filtering and three types of downsampling to construct a series of multi-scale examples. Then, the gradients of the loss function with respect to each scale are computed, and their average is used to determine the adversarial perturbations. The proposed SGP can be considered an input transformation with high extensibility that is easily integrated into most existing adversarial attacks. Extensive experiments demonstrate that in contrast to the state-of-the-art methods, SGP significantly enhances attack success rates against black-box defense models, with average attack success rates increasing by 2.3% to 32.6%, based only on transferability.

摘要: 对抗性示例的可移植性对深度神经网络构成了重大的安全挑战，深度神经网络可能会在不了解它们的情况下受到攻击。在本文中，我们提出了一种新的分段高斯金字塔（SGP）攻击方法来增强可移植性，特别是针对防御模型。与通常关注单尺度图像的现有方法不同，我们的方法采用高斯过滤和三种类型的下采样来构建一系列多尺度示例。然后，计算损失函数相对于每个尺度的梯度，并使用其平均值来确定对抗性扰动。提出的SGP可以被认为是具有高扩展性的输入转换，可以轻松集成到大多数现有的对抗攻击中。大量实验表明，与最先进的方法相比，SGP显着提高了针对黑匣子防御模型的攻击成功率，仅基于可移植性，平均攻击成功率增加了2.3%至32.6%。



## **2. Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training**

没有偷看的调整：LLM后培训的可证明隐私和泛化边界 cs.LG

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01752v1) [paper-pdf](http://arxiv.org/pdf/2507.01752v1)

**Authors**: Ismail Labiad, Mathurin Videau, Matthieu Kowalski, Marc Schoenauer, Alessandro Leite, Julia Kempe, Olivier Teytaud

**Abstract**: Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.

摘要: 基于对象的优化是深度学习的主力，通过反向传播提供高效且可扩展的训练。然而，它对大量标记数据的依赖引发了隐私和安全问题，例如容易受到数据中毒攻击和过度匹配的风险。相比之下，黑匣子优化方法将模型视为一个不透明的函数，仅依赖函数评估来指导优化，在数据访问受到限制、对抗风险较高或过度匹配令人担忧的场景中提供了一种有希望的替代方案。然而，黑匣子方法也带来了重大挑战，包括大型语言模型（LLM）中普遍存在的对多维参数空间的可扩展性较差，以及由于依赖大量模型评估而导致的高计算成本。本文介绍了BBoxER，这是一种用于LLM后训练的进化黑匣子方法，通过隐式压缩训练数据来引发信息瓶颈。利用信息流的可追溯性，我们在概括性、差异隐私、对数据中毒攻击的敏感性以及对提取攻击的鲁棒性方面提供了强大的理论界限。BBoxER在预先培训的LLM之上运行，除了非空洞的通用保证外，还提供适合在受限制或隐私敏感环境中部署的轻量级模块化增强。在LLM的实验中，我们经验地证明了Retrofit方法能够学习，展示了BBoxER的几次迭代如何提高性能并在推理数据集的基准上很好地概括。这使得BBoxER成为基于梯度的优化之上的一个有吸引力的附加组件。



## **3. Blockchain Address Poisoning**

区块链地址中毒 cs.CR

To appear in Proceedings of the 34th USENIX Security Symposium  (USENIX Security'25)

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2501.16681v3) [paper-pdf](http://arxiv.org/pdf/2501.16681v3)

**Authors**: Taro Tsuchiya, Jin-Dong Dong, Kyle Soska, Nicolas Christin

**Abstract**: In many blockchains, e.g., Ethereum, Binance Smart Chain (BSC), the primary representation used for wallet addresses is a hardly memorable 40-digit hexadecimal string. As a result, users often select addresses from their recent transaction history, which enables blockchain address poisoning. The adversary first generates lookalike addresses similar to one with which the victim has previously interacted, and then engages with the victim to ``poison'' their transaction history. The goal is to have the victim mistakenly send tokens to the lookalike address, as opposed to the intended recipient. Compared to contemporary studies, this paper provides four notable contributions. First, we develop a detection system and perform measurements over two years on both Ethereum and BSC. We identify 13~times more attack attempts than reported previously -- totaling 270M on-chain attacks targeting 17M victims. 6,633 incidents have caused at least 83.8M USD in losses, which makes blockchain address poisoning one of the largest cryptocurrency phishing schemes observed in the wild. Second, we analyze a few large attack entities using improved clustering techniques, and model attacker profitability and competition. Third, we reveal attack strategies -- targeted populations, success conditions (address similarity, timing), and cross-chain attacks. Fourth, we mathematically define and simulate the lookalike address generation process across various software- and hardware-based implementations, and identify a large-scale attacker group that appears to use GPUs. We also discuss defensive countermeasures.

摘要: 在许多区块链中，例如，以太坊，币安智能链（BSC），用于钱包地址的主要表示是一个几乎令人难忘的40位十六进制字符串。因此，用户经常从最近的交易历史记录中选择地址，这导致区块链地址中毒。对手首先生成与受害者之前互动过的地址相似的地址，然后与受害者互动以“毒害”他们的交易历史记录。目标是让受害者错误地将代币发送到外观相似的地址，而不是预期的收件人。与当代研究相比，本文提供了四个值得注意的贡献。首先，我们开发一个检测系统，并在两年内对以太坊和BSC进行测量。我们发现的攻击尝试比之前报告的多13倍--总计针对1700万受害者的2.7亿次链上攻击。6，633起事件已造成至少8，380万美元的损失，这使得区块链地址中毒成为野外观察到的最大的加密货币网络钓鱼计划之一。其次，我们使用改进的集群技术分析一些大型攻击实体，并对攻击者的盈利能力和竞争进行建模。第三，我们揭示了攻击策略--目标人群、成功条件（地址相似性、时机）和跨链攻击。第四，我们以数学方式定义和模拟各种基于软件和硬件的实现中相似的地址生成过程，并识别似乎使用图形处理器的大规模攻击者群体。我们还讨论防御对策。



## **4. Graph Representation-based Model Poisoning on Federated LLMs in CyberEdge Networks**

CyberEdge网络中联邦LLM上基于图表示的模型中毒 cs.CR

7 pages, 5 figures

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01694v1) [paper-pdf](http://arxiv.org/pdf/2507.01694v1)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) provide powerful generative capabilities in CyberEdge networks while protecting data privacy. However, FedLLMs remains highly vulnerable to model poisoning attacks. This article first reviews recent model poisoning techniques and existing defense mechanisms for FedLLMs, highlighting critical limitations, particularly under non-IID text distributions. In particular, current defenses primarily utilize distance-based outlier detection or norm constraints, operating under the assumption that adversarial updates significantly diverge from benign statistics. This assumption can fail when facing adaptive attackers targeting billionparameter LLMs. Next, this article investigates emerging Graph Representation-Based Model Poisoning (GRMP), a novel attack paradigm that leverages higher-order correlations among honest client gradients to synthesize malicious updates indistinguishable from legitimate model updates. GRMP can effectively evade advanced defenses, resulting in substantial accuracy loss and performance degradation. Moreover, this article outlines a research roadmap emphasizing the importance of graph-aware secure aggregation methods, FedLLMs-specific vulnerability metrics, and evaluation frameworks to strengthen the robustness of future federated language model deployments.

摘要: 联合大型语言模型（FedLLM）在CyberEdge网络中提供强大的生成能力，同时保护数据隐私。然而，FedLLM仍然极易受到模型中毒攻击。本文首先回顾了FedLLM最近的模型中毒技术和现有的防御机制，强调了关键的局限性，特别是在非IID文本分发下。特别是，当前的防御主要利用基于距离的离群值检测或规范约束，在对抗性更新与良性统计数据显着偏离的假设下运行。当面对针对十亿参数LLM的自适应攻击者时，这一假设可能会失败。接下来，本文研究了新兴的基于图表示的模型中毒（GRMP），这是一种新型攻击范式，它利用诚实客户端梯度之间的更高层相关性来合成与合法模型更新没有区别的恶意更新。GRMP可以有效规避高级防御，导致准确性大幅损失和性能下降。此外，本文还概述了一份研究路线图，强调图形感知的安全聚合方法、特定于FedLLM的漏洞指标和评估框架的重要性，以加强未来联邦语言模型部署的稳健性。



## **5. Learned-Database Systems Security**

学习数据库系统安全 cs.CR

Accepted at TMLR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2212.10318v4) [paper-pdf](http://arxiv.org/pdf/2212.10318v4)

**Authors**: Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, Nicolas Papernot

**Abstract**: A learned database system uses machine learning (ML) internally to improve performance. We can expect such systems to be vulnerable to some adversarial-ML attacks. Often, the learned component is shared between mutually-distrusting users or processes, much like microarchitectural resources such as caches, potentially giving rise to highly-realistic attacker models. However, compared to attacks on other ML-based systems, attackers face a level of indirection as they cannot interact directly with the learned model. Additionally, the difference between the attack surface of learned and non-learned versions of the same system is often subtle. These factors obfuscate the de-facto risks that the incorporation of ML carries. We analyze the root causes of potentially-increased attack surface in learned database systems and develop a framework for identifying vulnerabilities that stem from the use of ML. We apply our framework to a broad set of learned components currently being explored in the database community. To empirically validate the vulnerabilities surfaced by our framework, we choose 3 of them and implement and evaluate exploits against these. We show that the use of ML cause leakage of past queries in a database, enable a poisoning attack that causes exponential memory blowup in an index structure and crashes it in seconds, and enable index users to snoop on each others' key distributions by timing queries over their own keys. We find that adversarial ML is an universal threat against learned components in database systems, point to open research gaps in our understanding of learned-systems security, and conclude by discussing mitigations, while noting that data leakage is inherent in systems whose learned component is shared between multiple parties.

摘要: 学习数据库系统在内部使用机器学习（ML）来提高性能。我们可以预计此类系统很容易受到一些对抗ML攻击。通常，学习到的组件在相互不信任的用户或流程之间共享，就像缓存等微架构资源一样，可能会产生高度真实的攻击者模型。然而，与对其他基于ML的系统的攻击相比，攻击者面临一定程度的间接性，因为他们无法与学习模型直接交互。此外，同一系统的学习版本和非学习版本的攻击表面之间的差异通常很微妙。这些因素混淆了合并ML所带来的事实风险。我们分析了学习数据库系统中攻击面可能增加的根本原因，并开发了一个框架来识别源于ML的使用的漏洞。我们将我们的框架应用于数据库社区目前正在探索的一系列广泛的学习组件。为了从经验上验证我们的框架中出现的漏洞，我们选择了其中3个漏洞，并针对这些漏洞实施和评估利用。我们表明，ML的使用会导致数据库中过去的查询泄露，引发中毒攻击，导致索引结构中的指数级内存爆炸并在几秒钟内使其崩溃，并使索引用户能够通过对自己的密钥进行计时来窥探彼此的密钥分布。我们发现对抗性ML是针对数据库系统中学习组件的普遍威胁，指出了我们对学习系统安全性的理解中存在的研究差距，并通过讨论缓解措施来得出结论，同时指出数据泄露是其学习组件在多方之间共享的系统中固有的。



## **6. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

插槽：通过图强化学习进行源驱动APT检测 cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2410.17910v3) [paper-pdf](http://arxiv.org/pdf/2410.17910v3)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.

摘要: 高级持续性威胁（APT）代表复杂的网络攻击，其特征是能够在受害者系统中长时间不被发现，旨在泄露敏感数据或扰乱运营。现有的检测方法常常难以有效识别这些复杂的威胁、构建防御促进的攻击链或抵抗对抗性攻击。为了克服这些挑战，我们提出了Slot，这是一种基于出处图和图强化学习的高级APT检测方法。Slot擅长通过出处图挖掘发现系统行为之间的多层隐藏关系，例如因果关系、上下文关系和间接联系。通过开创图强化学习的集成，Slot动态适应新的用户活动和不断发展的攻击策略，增强其对对抗性攻击的弹性。此外，Slot还根据检测到的攻击，通过集群算法自动构建攻击链，提供攻击路径的精确识别，促进防御策略的制定。对现实世界数据集的评估证明了Slot在APT检测方面出色的准确性、效率、适应性和稳健性，大多数指标都超越了最先进的方法。此外，为评估Slot支持APT防御的有效性而进行的案例研究进一步确立了其作为网络安全保护实用且可靠的工具的地位。



## **7. DARTS: A Dual-View Attack Framework for Targeted Manipulation in Federated Sequential Recommendation**

DARTS：联邦顺序推荐中针对性操纵的双视图攻击框架 cs.IR

10 pages. arXiv admin note: substantial text overlap with  arXiv:2409.07500; text overlap with arXiv:2212.05399 by other authors

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01383v1) [paper-pdf](http://arxiv.org/pdf/2507.01383v1)

**Authors**: Qitao Qin, Yucong Luo, Zhibo Chu

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation(FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models. Our codes are publicly available.

摘要: 联合推荐（FedRec）通过支持个性化模型的去中心化训练来保护用户隐私，但这种架构本质上很容易受到对抗攻击。出于商业和社会影响考虑，人们对FedRec系统中的定向攻击进行了大量研究。然而，这项工作的大部分内容在很大程度上忽视了推荐模型的差异稳健性。此外，我们的经验研究结果表明，现有的有针对性的攻击方法在联合顺序推荐（FSR）任务中仅实现有限的有效性。在这些观察的推动下，我们专注于调查FSR中的有针对性的攻击，并提出了一种新颖的双视图攻击框架，名为DV-FSR。这种攻击方法独特地将基于采样的显式策略与基于对比学习的隐式梯度策略结合起来，以协调一致的攻击。此外，我们还引入了一种针对FSR中的定向攻击量身定制的特定防御机制，旨在评估我们提出的攻击方法的缓解效果。大量实验验证了我们提出的方法对代表性序列模型的有效性。我们的代码是公开的。



## **8. 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation**

3D高斯飞溅驱动的多视图鲁棒物理对抗伪装生成 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01367v1) [paper-pdf](http://arxiv.org/pdf/2507.01367v1)

**Authors**: Tianrui Lou, Xiaojun Jia, Siyuan Liang, Jiawei Liang, Ming Zhang, Yanjun Xiao, Xiaochun Cao

**Abstract**: Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.

摘要: 物理对抗攻击方法暴露了深度神经网络的漏洞，并对自动驾驶等安全关键场景构成重大威胁。与基于补丁的攻击相比，基于伪装的物理攻击是一种更有前途的方法，可以在复杂的物理环境中提供更强的对抗效果。然而，大多数先前的工作依赖于目标对象的网格先验和模拟器构建的虚拟环境，获取这些先验很耗时，并且不可避免地与现实世界不同。此外，由于训练图像中背景的限制，以前的方法常常无法产生多视图鲁棒的对抗伪装，并且往往会陷入次优解决方案。由于这些原因，之前的工作缺乏针对不同观点和物理环境的对抗有效性和稳健性。我们提出了一种基于3D高斯飞溅（3DGS）的物理攻击框架，名为PGA，该框架只需少量图像即可提供快速精确的重建，并具有照片真实感的渲染能力。我们的框架通过防止高斯之间的相互遮挡和自遮挡，并采用调整每个观点的成像背景的最小-最大优化方法，帮助算法过滤掉非鲁棒的对抗特征，进一步增强了交叉视图的鲁棒性和对抗有效性。大量实验验证了PGA的有效性和优越性。我们的代码可访问：https://github.com/TRLou/PGA。



## **9. Backdooring Bias (B^2) into Stable Diffusion Models**

稳定扩散模型中的后备偏差（B#2） cs.LG

Accepted to USENIX Security '25

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2406.15213v3) [paper-pdf](http://arxiv.org/pdf/2406.15213v3)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasaryan, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional diffusion models have revolutionized image generation by enabling users to create realistic, high-quality images from textual prompts, significantly enhancing artistic creation and visual communication. However, these advancements also introduce an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence public opinion and spread propaganda. In this paper, we study an attack vector that allows an adversary to inject arbitrary bias into a target model. The attack leverages low-cost backdooring techniques using a targeted set of natural textual triggers embedded within a small number of malicious data samples produced with public generative models. An adversary could pick common sequences of words that can then be inadvertently activated by benign users during inference. We investigate the feasibility and challenges of such attacks, demonstrating how modern generative models have made this adversarial process both easier and more adaptable. On the other hand, we explore various aspects of the detectability of such attacks and demonstrate that the model's utility remains intact in the absence of the triggers. Our extensive experiments using over 200,000 generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases maintain strong text-image alignment, highlighting the challenges in detecting biased images without knowing that bias in advance. Our cost analysis confirms the low financial barrier ($10-$15) to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in diffusion models.

摘要: 大型文本条件扩散模型的最新进展使用户能够根据文本提示创建真实、高质量的图像，从而彻底改变了图像生成，从而显着增强了艺术创作和视觉传达。然而，这些进步也带来了一个未充分探索的攻击机会：对手出于恶意意图将偏见引入生成的图像的可能性，例如，影响舆论并传播宣传。在本文中，我们研究了允许对手将任意偏差注入目标模型的攻击载体。该攻击利用低成本后门技术，使用嵌入公共生成模型生成的少量恶意数据样本中的一组有针对性的自然文本触发器。对手可能会选择常见的单词序列，然后在推理过程中被良性用户无意中激活。我们调查此类攻击的可行性和挑战，展示现代生成模型如何使这种对抗过程变得更容易、更适应。另一方面，我们探索了此类攻击可检测性的各个方面，并证明在没有触发器的情况下，该模型的实用性仍然完好无损。我们使用超过200，000张生成的图像并针对数百个微调模型进行了广泛的实验，证明了所提出的后门攻击的可行性。我们说明了这些偏见如何保持文本与图像的强对齐，强调了在事先不知道偏见的情况下检测偏见图像的挑战。我们的成本分析证实了执行此类攻击的财务障碍较低（10 - 15美元），强调了针对扩散模型中此类漏洞的强大防御策略的必要性。



## **10. ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks**

ICLShield：探索和缓解上下文学习后门攻击 cs.LG

ICML 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01321v1) [paper-pdf](http://arxiv.org/pdf/2507.01321v1)

**Authors**: Zhiyao Ren, Siyuan Liang, Aishan Liu, Dacheng Tao

**Abstract**: In-context learning (ICL) has demonstrated remarkable success in large language models (LLMs) due to its adaptability and parameter-free nature. However, it also introduces a critical vulnerability to backdoor attacks, where adversaries can manipulate LLM behaviors by simply poisoning a few ICL demonstrations. In this paper, we propose, for the first time, the dual-learning hypothesis, which posits that LLMs simultaneously learn both the task-relevant latent concepts and backdoor latent concepts within poisoned demonstrations, jointly influencing the probability of model outputs. Through theoretical analysis, we derive an upper bound for ICL backdoor effects, revealing that the vulnerability is dominated by the concept preference ratio between the task and the backdoor. Motivated by these findings, we propose ICLShield, a defense mechanism that dynamically adjusts the concept preference ratio. Our method encourages LLMs to select clean demonstrations during the ICL phase by leveraging confidence and similarity scores, effectively mitigating susceptibility to backdoor attacks. Extensive experiments across multiple LLMs and tasks demonstrate that our method achieves state-of-the-art defense effectiveness, significantly outperforming existing approaches (+26.02% on average). Furthermore, our method exhibits exceptional adaptability and defensive performance even for closed-source models (e.g., GPT-4).

摘要: 上下文学习（ICL）因其适应性和无参数性质而在大型语言模型（LLM）中取得了显着的成功。然而，它也引入了后门攻击的关键漏洞，对手可以通过简单地毒害一些ICL演示来操纵LLM行为。在本文中，我们首次提出了双重学习假设，该假设LLM同时学习与任务相关的潜在概念和中毒演示中的后门潜在概念，共同影响模型输出的可能性。通过理论分析，我们推导出ICL后门效应的上界，揭示了漏洞由任务与后门之间的概念偏好比决定。受这些发现的启发，我们提出了ICLShield，这是一种动态调整概念偏好比的防御机制。我们的方法鼓励LLM通过利用置信度和相似性分数在ICL阶段选择干净的演示，从而有效地降低对后门攻击的敏感性。跨多个LLM和任务的广泛实验表明，我们的方法实现了最先进的防御有效性，显着优于现有方法（平均+26.02%）。此外，即使对于闭源模型（例如，GPT-4）。



## **11. Defensive Adversarial CAPTCHA: A Semantics-Driven Framework for Natural Adversarial Example Generation**

防御性对抗验证码：用于自然对抗示例生成的语义驱动框架 cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2506.10685v3) [paper-pdf](http://arxiv.org/pdf/2506.10685v3)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Cong Wu, Tao Li, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: Traditional CAPTCHA (Completely Automated Public Turing Test to Tell Computers and Humans Apart) schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on the original image characteristics, resulting in distortions that hinder human interpretation and limit their applicability in scenarios where no initial input images are available. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (DAC), a novel framework that generates high-fidelity adversarial examples guided by attacker-specified semantics information. Leveraging a Large Language Model (LLM), DAC enhances CAPTCHA diversity and enriches the semantic information. To address various application scenarios, we examine the white-box targeted attack scenario and the black box untargeted attack scenario. For target attacks, we introduce two latent noise variables that are alternately guided in the diffusion step to achieve robust inversion. The synergy between gradient guidance and latent variable optimization achieved in this way ensures that the generated adversarial examples not only accurately align with the target conditions but also achieve optimal performance in terms of distributional consistency and attack effectiveness. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-DAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show that the defensive adversarial CAPTCHA generated by BP-DAC is able to defend against most of the unknown models, and the generated CAPTCHA is indistinguishable to both humans and DNNs.

摘要: 传统的CAPTCHA（完全自动化公共图灵测试来区分计算机和人类）计划越来越容易受到深度神经网络（DNN）支持的自动化攻击。现有的对抗攻击方法通常依赖于原始图像特征，从而导致失真，阻碍人类解释并限制其在没有初始输入图像可用的场景中的适用性。为了解决这些挑战，我们提出了无源对抗性验证码（ADC），这是一种新颖的框架，可以在攻击者指定的语义信息的指导下生成高保真对抗性示例。利用大型语言模型（LLM），DEC增强了CAPTCHA的多样性并丰富了语义信息。为了应对各种应用场景，我们研究了白盒定向攻击场景和黑匣子非定向攻击场景。对于目标攻击，我们引入了两个潜在噪音变量，它们在扩散步骤中交替引导，以实现鲁棒的反转。通过这种方式实现的梯度引导和潜在变量优化之间的协同作用，确保生成的对抗示例不仅与目标条件准确对齐，而且在分布一致性和攻击有效性方面实现最佳性能。在无目标攻击中，特别是对于黑匣子场景，我们引入了双路径无源对抗性CAPTCHA（BP-ADC），这是一种两步优化策略，采用多峰梯度和双路径优化来实现高效的误分类。实验表明，BP-ADC生成的防御性对抗CAPTCHA能够防御大多数未知模型，并且生成的CAPTCHA对于人类和DNN来说都无法区分。



## **12. CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs**

CAWLRY-V：一个用于视频MLLM对抗性攻击的大规模生成器框架 cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00817v1) [paper-pdf](http://arxiv.org/pdf/2507.00817v1)

**Authors**: Jiaming Zhang, Rui Hu, Qing Guo, Wei Yang Bryan Lim

**Abstract**: Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems.

摘要: 视频多模式大型语言模型（V-MLLM）在时态推理和跨模式理解方面表现出了令人印象深刻的能力，但由于独特的挑战，它们对对抗性攻击的脆弱性仍然没有得到充分的研究：复杂的跨模式推理机制、时态依赖性和计算限制。我们提出了CAWLRY-V（跨模式视觉对抗屈服视频），这是一个新颖的框架，直接针对V-MLLM中视觉感知和语言生成之间的关键界面。我们的方法引入了两个关键创新：（1）双目标语义视觉损失函数，它同时扰乱模型的文本生成日志和视觉表示以破坏跨模式集成，以及（2）计算高效的两阶段生成器框架，它将跨模型可移植性的大规模预训练与时空一致性的专门微调相结合。对全面视频理解基准的实证评估表明，CAWLRY-V的表现显着优于现有的攻击方法，比商业系统（GPT-4.1、Gemini 2.0）和开源模型（QwenVL-2.5、InternVL-2.5、Llava-Video、Aria、MiniCPM-o-2.6）的最佳基线攻击平均改进了22.8%。我们的框架通过隐式时间一致性建模而不是显式正规化来实现灵活性，即使在图像理解方面也能显着提高性能（平均提高34.4%）。这一能力展示了CAWLRY-V作为跨多模式系统对抗性研究的基础方法的潜力。



## **13. Cage-Based Deformation for Transferable and Undefendable Point Cloud Attack**

基于笼子的变形可转移和不可发现的点云攻击 cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00690v1) [paper-pdf](http://arxiv.org/pdf/2507.00690v1)

**Authors**: Keke Tang, Ziyong Du, Weilong Peng, Xiaofei Wang, Peican Zhu, Ligang Liu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds often impose strict geometric constraints to preserve plausibility; however, such constraints inherently limit transferability and undefendability. While deformation offers an alternative, existing unstructured approaches may introduce unnatural distortions, making adversarial point clouds conspicuous and undermining their plausibility. In this paper, we propose CageAttack, a cage-based deformation framework that produces natural adversarial point clouds. It first constructs a cage around the target object, providing a structured basis for smooth, natural-looking deformation. Perturbations are then applied to the cage vertices, which seamlessly propagate to the point cloud, ensuring that the resulting deformations remain intrinsic to the object and preserve plausibility. Extensive experiments on seven 3D deep neural network classifiers across three datasets show that CageAttack achieves a superior balance among transferability, undefendability, and plausibility, outperforming state-of-the-art methods. Codes will be made public upon acceptance.

摘要: 对点云的对抗攻击通常会施加严格的几何约束以保持相似性;然而，此类约束本质上限制了可移植性和不可分割性。虽然变形提供了一种替代方案，但现有的非结构化方法可能会引入不自然的扭曲，使对抗性点云变得明显并破坏其合理性。在本文中，我们提出了CageAttack，这是一个基于笼子的变形框架，可以产生自然的对抗点云。它首先在目标物体周围构建一个笼子，为光滑、自然的变形提供结构化基础。然后将扰动应用于笼形点，其无缝传播到点云，确保产生的变形保持对象固有并保持相似性。对三个数据集的七个3D深度神经网络分类器进行的广泛实验表明，CageAttack在可移植性、不可预测性和可信性之间实现了卓越的平衡，优于最先进的方法。代码在接受后将公开。



## **14. How Resilient is QUIC to Security and Privacy Attacks?**

QUIC对安全和隐私攻击的弹性如何？ cs.CR

7 pages, 1 figure, 1 table

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2401.06657v3) [paper-pdf](http://arxiv.org/pdf/2401.06657v3)

**Authors**: Jayasree Sengupta, Debasmita Dey, Simone Ferlin-Reiter, Nirnay Ghosh, Vaibhav Bajpai

**Abstract**: QUIC has rapidly evolved into a cornerstone transport protocol for secure, low-latency communications, yet its deployment continues to expose critical security and privacy vulnerabilities, particularly during connection establishment phases and via traffic analysis. This paper systematically revisits a comprehensive set of attacks on QUIC and emerging privacy threats. Building upon these observations, we critically analyze recent IETF mitigation efforts, including TLS Encrypted Client Hello (ECH), Oblivious HTTP (OHTTP) and MASQUE. We analyze how these mechanisms enhance privacy while introducing new operational risks, particularly under adversarial load. Additionally, we discuss emerging challenges posed by post-quantum cryptographic (PQC) handshakes, including handshake expansion and metadata leakage risks. Our analysis highlights ongoing gaps between theoretical defenses and practical deployments, and proposes new research directions focused on adaptive privacy mechanisms. Building on these insights, we propose future directions to ensure long-term security of QUIC and aim to guide its evolution as a robust, privacy-preserving, and resilient transport foundation for the next-generation Internet.

摘要: QUIC已迅速发展成为安全、低延迟通信的基石传输协议，但其部署继续暴露关键的安全和隐私漏洞，特别是在连接建立阶段和流量分析期间。本文系统性地重新审视了针对QUIC和新出现的隐私威胁的一系列全面攻击。在这些观察的基础上，我们批判性地分析了最近的ETF缓解工作，包括SSL加密客户端Hello（ECH）、不经意的HTTP（Ohttp）和MASQUE。我们分析了这些机制如何增强隐私，同时引入新的运营风险，特别是在对抗负载下。此外，我们还讨论了后量子加密（PQC）握手带来的新挑战，包括握手扩展和元数据泄露风险。我们的分析强调了理论防御和实际部署之间持续存在的差距，并提出了专注于自适应隐私机制的新研究方向。在这些见解的基础上，我们提出了未来方向，以确保QUIC的长期安全，并旨在引导其发展成为下一代互联网的强大、保护隐私和弹性传输基础。



## **15. Lazarus Group Targets Crypto-Wallets and Financial Data while employing new Tradecrafts**

Lazarus Group瞄准加密钱包和金融数据，同时采用新的Tradecrafts cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2505.21725v2) [paper-pdf](http://arxiv.org/pdf/2505.21725v2)

**Authors**: Alessio Di Santo

**Abstract**: This report presents a comprehensive analysis of a malicious software sample, detailing its architecture, behavioral characteristics, and underlying intent. Through static and dynamic examination, the malware core functionalities, including persistence mechanisms, command-and-control communication, and data exfiltration routines, are identified and its supporting infrastructure is mapped. By correlating observed indicators of compromise with known techniques, tactics, and procedures, this analysis situates the sample within the broader context of contemporary threat campaigns and infers the capabilities and motivations of its likely threat actor.   Building on these findings, actionable threat intelligence is provided to support proactive defenses. Threat hunting teams receive precise detection hypotheses for uncovering latent adversarial presence, while monitoring systems can refine alert logic to detect anomalous activity in real time. Finally, the report discusses how this structured intelligence enhances predictive risk assessments, informs vulnerability prioritization, and strengthens organizational resilience against advanced persistent threats. By integrating detailed technical insights with strategic threat landscape mapping, this malware analysis report not only reconstructs past adversary actions but also establishes a robust foundation for anticipating and mitigating future attacks.

摘要: 本报告对恶意软件样本进行了全面分析，详细介绍了其架构、行为特征和潜在意图。通过静态和动态检查，识别恶意软件的核心功能，包括持久性机制、命令和控制通信以及数据溢出例程，并绘制其支持基础设施。通过将观察到的妥协指标与已知的技术、策略和程序关联起来，该分析将样本置于当代威胁活动的更广泛背景下，并推断其可能的威胁参与者的能力和动机。   在这些发现的基础上，提供可操作的威胁情报来支持主动防御。威胁搜寻团队接收精确的检测假设，以发现潜在的对抗存在，而监控系统可以完善警报逻辑以实时检测异常活动。最后，该报告讨论了这种结构化智能如何增强预测性风险评估、为漏洞优先排序提供信息以及加强组织应对高级持续威胁的弹性。通过将详细的技术见解与战略威胁格局映射集成，该恶意软件分析报告不仅重建了过去的对手行为，还为预测和减轻未来的攻击奠定了坚实的基础。



## **16. Plug. Play. Persist. Inside a Ready-to-Go Havoc C2 Infrastructure**

插头。玩吧坚持。准备就绪的Havoc C2基础设施内部 cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2507.00189v1) [paper-pdf](http://arxiv.org/pdf/2507.00189v1)

**Authors**: Alessio Di Santo

**Abstract**: This analysis focuses on a single Azure-hosted Virtual Machine at 52.230.23.114 that the adversary converted into an all-in-one delivery, staging and Command-and-Control node. The host advertises an out-of-date Apache 2.4.52 instance whose open directory exposes phishing lures, PowerShell loaders, Reflective Shell-Code, compiled Havoc Demon implants and a toolbox of lateral-movement binaries; the same server also answers on 8443/80 for encrypted beacon traffic. The web tier is riddled with publicly documented critical vulnerabilities, that would have allowed initial code-execution had the attackers not already owned the device.   Initial access is delivered through an HTML file that, once de-obfuscated, perfectly mimics Google Unusual sign-in attempt notification and funnels victims toward credential collection. A PowerShell command follows: it disables AMSI in-memory, downloads a Base64-encoded stub, allocates RWX pages and starts the shell-code without ever touching disk. That stub reconstructs a DLL in memory using the Reflective-Loader technique and hands control to Havoc Demon implant. Every Demon variant-32- and 64-bit alike-talks to the same backend, resolves Windows APIs with hashed look-ups, and hides its activity behind indirect syscalls.   Runtime telemetry shows interests in registry under Image File Execution Options, deliberate queries to Software Restriction Policy keys, and heavy use of Crypto DLLs to protect payloads and C2 traffic. The attacker toolkit further contains Chisel, PsExec, Doppelganger and Whisker, some of them re-compiled under user directories that leak the developer personas tonzking123 and thobt. Collectively the findings paint a picture of a technically adept actor who values rapid re-tooling over deep operational security, leaning on Havoc modularity and on legitimate cloud services to blend malicious flows into ordinary enterprise traffic.

摘要: 此分析重点关注52.230.23.114上的单个Azure托管虚拟机，对手将其转换为一体化交付、中转和命令与控制节点。主机通知一个过时的Apache 2.4.52实例，其打开目录暴露了网络钓鱼诱饵、PowerShell加载器、反射性Shell-Code、已编译的Havoc Demon植入物和侧向移动二进制文件工具箱;同一服务器还在8443/80上回答加密信标流量。Web层充满了公开记录的关键漏洞，如果攻击者尚未拥有该设备，这些漏洞将允许初始代码执行。   初始访问通过一个HTML文件进行，该文件一旦去模糊，就会完美模仿Google Unusual登录尝试通知，并将受害者引导到凭证收集中。下面是一个Shell命令：它禁用内存中的AMSI，下载Base 64编码的树桩，分配RWX页面并在不接触磁盘的情况下启动shell代码。该树桩使用反射加载器技术在内存中重建了一个动态链接库，并将控制权交给Havoc Demon植入物。每个Demon变体（32位和64位相似）都与相同的后台对话，通过哈希查找来解析Windows API，并将其活动隐藏在间接系统缩放后面。   收件箱遥感显示对图像文件执行选项下的注册表的兴趣、对软件限制策略密钥的故意查询以及大量使用加密DLC来保护有效负载和C2流量。攻击者工具包还包含Chisel、PsExec、Doppelganger和Whisker，其中一些是在泄露开发人员角色tonzking 123和thobt的用户目录下重新编译的。总的来说，这些调查结果描绘了一幅技术精湛的参与者的图景，他重视快速重组而不是深度运营安全，依靠Havoc模块化和合法的云服务将恶意流量混合到普通企业流量中。



## **17. SQUASH: A SWAP-Based Quantum Attack to Sabotage Hybrid Quantum Neural Networks**

SQUASH：一种基于交换的量子攻击，旨在破坏混合量子神经网络 quant-ph

Keywords: Quantum Machine Learning, Hybrid Quantum Neural Networks,  SWAP Test, Fidelity, Circuit-level Attack

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24081v1) [paper-pdf](http://arxiv.org/pdf/2506.24081v1)

**Authors**: Rahul Kumar, Wenqi Wei, Ying Mao, Junaid Farooq, Ying Wang, Juntao Chen

**Abstract**: We propose a circuit-level attack, SQUASH, a SWAP-Based Quantum Attack to sabotage Hybrid Quantum Neural Networks (HQNNs) for classification tasks. SQUASH is executed by inserting SWAP gate(s) into the variational quantum circuit of the victim HQNN. Unlike conventional noise-based or adversarial input attacks, SQUASH directly manipulates the circuit structure, leading to qubit misalignment and disrupting quantum state evolution. This attack is highly stealthy, as it does not require access to training data or introduce detectable perturbations in input states. Our results demonstrate that SQUASH significantly degrades classification performance, with untargeted SWAP attacks reducing accuracy by up to 74.08\% and targeted SWAP attacks reducing target class accuracy by up to 79.78\%. These findings reveal a critical vulnerability in HQNN implementations, underscoring the need for more resilient architectures against circuit-level adversarial interventions.

摘要: 我们提出了一种电路级攻击SQUASH，一种基于SWAP的量子攻击，以破坏用于分类任务的混合量子神经网络（HQNN）。SQUASH通过将SWAP门插入到受害者HQNN的变分量子电路中来执行。与传统的基于噪声或对抗性输入攻击不同，SQUASH直接操纵电路结构，导致量子比特错位并破坏量子态演化。这种攻击是高度隐蔽的，因为它不需要访问训练数据或在输入状态中引入可检测的扰动。我们的结果表明，SQUASH会显着降低分类性能，非目标SWAP攻击将准确性降低高达74.08%，而目标SWAP攻击将目标类准确性降低高达79.78%。这些发现揭示了HQNN实现中的一个关键漏洞，强调了对电路级对抗干预的更具弹性的架构的需要。



## **18. STACK: Adversarial Attacks on LLM Safeguard Pipelines**

STACK：对LLM Safeguard Pipelines的对抗性攻击 cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24068v1) [paper-pdf](http://arxiv.org/pdf/2506.24068v1)

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.

摘要: 前沿人工智能开发人员依靠多层保障措施来防止人工智能系统的灾难性滥用。Anthropic使用这样的防御管道来保护他们最新的Claude 4 Opus模型，包括Google DeepMind和OpenAI在内的其他前沿开发商承诺很快部署类似的防御。然而，此类管道的安全性尚不清楚，之前评估或攻击这些管道的工作有限。我们通过开发和组建开源防御管道来解决这一差距。首先，我们发现一种新型的几次激发输入和输出分类器在三次攻击和两个数据集中优于最先进的开权保护模型ShieldGemma，将灾难性滥用数据集ClearHarm的攻击成功率（ASO）降低至0%。其次，我们引入了一个STaged AttaCK（STACK）过程，该过程在ClearHarm上实现了71%的ASB，针对少量镜头提示的分类器管道进行黑匣子攻击。最后，我们还在传输环境中评估了STACK，实现了33%的ASB，提供了初步证据，证明在不访问目标管道的情况下设计攻击是可行的。最后，我们建议开发人员可以用来阻止分阶段攻击的具体缓解措施。



## **19. Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies**

针对闭箱对抗攻击的基于假设的优化以及与进化策略的联系 math.OC

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24048v1) [paper-pdf](http://arxiv.org/pdf/2506.24048v1)

**Authors**: Tim Roith, Leon Bungert, Philipp Wacker

**Abstract**: Consensus-based optimization (CBO) has established itself as an efficient gradient-free optimization scheme, with attractive mathematical properties, such as mean-field convergence results for non-convex loss functions. In this work, we study CBO in the context of closed-box adversarial attacks, which are imperceptible input perturbations that aim to fool a classifier, without accessing its gradient. Our contribution is to establish a connection between the so-called consensus hopping as introduced by Riedl et al. and natural evolution strategies (NES) commonly applied in the context of adversarial attacks and to rigorously relate both methods to gradient-based optimization schemes. Beyond that, we provide a comprehensive experimental study that shows that despite the conceptual similarities, CBO can outperform NES and other evolutionary strategies in certain scenarios.

摘要: 基于边界的优化（CBO）已经成为一种高效的无梯度优化方案，具有吸引人的数学性质，例如非凸损失函数的平均场收敛结果。在这项工作中，我们在闭箱对抗攻击的背景下研究CBO，这是一种难以察觉的输入扰动，旨在欺骗分类器，而无需访问其梯度。我们的贡献是在Riedl等人提出的所谓共识跳跃与通常在对抗性攻击背景下应用的自然进化策略（NES）之间建立联系，并将这两种方法与基于梯度的优化方案严格联系起来。除此之外，我们还提供了一项全面的实验研究，表明尽管概念相似，但CBO在某些情况下可以优于NES和其他进化策略。



## **20. Quickest Detection of Adversarial Attacks Against Correlated Equilibria**

最快检测针对相关均衡的对抗攻击 cs.GT

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24040v1) [paper-pdf](http://arxiv.org/pdf/2506.24040v1)

**Authors**: Kiarash Kazari, Aris Kanellopoulos, György Dán

**Abstract**: We consider correlated equilibria in strategic games in an adversarial environment, where an adversary can compromise the public signal used by the players for choosing their strategies, while players aim at detecting a potential attack as soon as possible to avoid loss of utility. We model the interaction between the adversary and the players as a zero-sum game and we derive the maxmin strategies for both the defender and the attacker using the framework of quickest change detection. We define a class of adversarial strategies that achieve the optimal trade-off between attack impact and attack detectability and show that a generalized CUSUM scheme is asymptotically optimal for the detection of the attacks. Our numerical results on the Sioux-Falls benchmark traffic routing game show that the proposed detection scheme can effectively limit the utility loss by a potential adversary.

摘要: 我们考虑了在对抗环境中的策略博弈中的相关均衡，在这种环境中，对手可以损害玩家选择策略所使用的公共信号，而玩家的目标是尽快检测到潜在的攻击，以避免效用损失。我们建模的对手和球员之间的相互作用作为一个零和博弈，我们推导出最大最小的防御者和攻击者使用的框架，最快的变化检测的策略。我们定义了一类对抗策略，实现了攻击影响和攻击可检测性之间的最佳权衡，并证明了广义的CRAMUM方案对于检测攻击是渐近最优的。在Sioux-Falls基准流量路由博弈上的数值结果表明，该检测方案可以有效地限制潜在对手的效用损失。



## **21. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

谜语我！检索增强一代的隐形会员推断 cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够通过利用外部知识数据库来生成接地响应，而无需更改模型参数。尽管缺乏权重调整可以防止模型参数泄露，但它引入了推理对手利用模型上下文中检索到的文档的风险。现有的隶属关系推断和数据提取方法通常依赖于越狱或精心制作的非自然查询，这些查询可以通过RAG系统中常见的查询重写技术轻松检测或阻止。在这项工作中，我们介绍了审讯攻击（IA），这是一种针对RAG收件箱中文档的成员资格推断技术。通过制作仅在目标文档存在的情况下才能回答的自然文本查询，我们的方法仅用30个查询就能证明成功推理，同时保持隐蔽性;简单的检测器识别来自现有方法的对抗性提示的频率高达约76倍，比我们的攻击产生的提示。我们观察到，在各种RAG配置中，TPR@1%FPR比之前的推理攻击提高了2倍，同时每个文档推理的成本不到0.02美元。



## **22. Benchmarking Spiking Neural Network Learning Methods with Varying Locality**

对具有不同局部性的尖峰神经网络学习方法进行基准测试 cs.NE

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2402.01782v2) [paper-pdf](http://arxiv.org/pdf/2402.01782v2)

**Authors**: Jiaqi Lin, Sen Lu, Malyaban Bal, Abhronil Sengupta

**Abstract**: Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have been shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but come with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, given the implicitly recurrent nature of SNNs, this research investigates the influence of the addition of explicit recurrence to SNNs. We experimentally prove that the addition of explicit recurrent weights enhances the robustness of SNNs. We also investigate the performance of local learning methods under gradient and non-gradient-based adversarial attacks.

摘要: 尖峰神经网络（SNN）提供了更真实的神经元动力学，已被证明在几项机器学习任务中可以实现与人工神经网络（ANN）相当的性能。信息在基于事件的机制中作为SNN内的尖峰进行处理，从而显着降低了能源消耗。然而，由于尖峰机制的不可微性质，训练SNN具有挑战性。传统方法，例如时间反向传播（BPTT），已经显示出有效性，但会带来额外的计算和存储成本，并且在生物学上是不可信的。相比之下，最近的作品提出了具有不同局部性的替代学习方法，证明了分类任务的成功。在这项工作中，我们表明这些方法在训练过程中有相似之处，同时它们在生物相似性和性能之间进行了权衡。此外，鉴于SNN的隐式回归性质，本研究调查了SNN添加显式回归的影响。我们通过实验证明，添加显式循环权重增强了SNN的鲁棒性。我们还研究了本地学习方法在梯度和非基于梯度的对抗攻击下的性能。



## **23. A Unified Framework for Stealthy Adversarial Generation via Latent Optimization and Transferability Enhancement**

通过潜在优化和可移植性增强实现隐形对抗生成的统一框架 cs.CV

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23676v1) [paper-pdf](http://arxiv.org/pdf/2506.23676v1)

**Authors**: Gaozheng Pei, Ke Ma, Dongpeng Zhang, Chengzhi Sun, Qianqian Xu, Qingming Huang

**Abstract**: Due to their powerful image generation capabilities, diffusion-based adversarial example generation methods through image editing are rapidly gaining popularity. However, due to reliance on the discriminative capability of the diffusion model, these diffusion-based methods often struggle to generalize beyond conventional image classification tasks, such as in Deepfake detection. Moreover, traditional strategies for enhancing adversarial example transferability are challenging to adapt to these methods. To address these challenges, we propose a unified framework that seamlessly incorporates traditional transferability enhancement strategies into diffusion model-based adversarial example generation via image editing, enabling their application across a wider range of downstream tasks. Our method won first place in the "1st Adversarial Attacks on Deepfake Detectors: A Challenge in the Era of AI-Generated Media" competition at ACM MM25, which validates the effectiveness of our approach.

摘要: 由于其强大的图像生成能力，通过图像编辑的基于扩散的对抗性示例生成方法正在迅速流行。然而，由于依赖于扩散模型的辨别能力，这些基于扩散的方法通常很难概括超出传统图像分类任务，例如Deepfake检测。此外，增强对抗性示例可移植性的传统策略在适应这些方法方面具有挑战性。为了应对这些挑战，我们提出了一个统一的框架，通过图像编辑将传统的可移植性增强策略无缝地整合到基于扩散模型的对抗性示例生成中，使其能够在更广泛的下游任务中应用。我们的方法在ACN MM25举行的“Deepfake Detector的第一次对抗攻击：人工智能生成媒体时代的挑战”竞赛中获得了第一名，这验证了我们方法的有效性。



## **24. Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack**

错误信息分类系统对BeamAttack对抗示例的鲁棒性 cs.CL

12 pages main text, 27 pages total including references and  appendices. 13 figures, 10 tables. Accepted for publication in the LNCS  proceedings of CLEF 2025 (Best-of-Labs track)

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23661v1) [paper-pdf](http://arxiv.org/pdf/2506.23661v1)

**Authors**: Arnisa Fazla, Lucas Krauter, David Guzman Piedrahita, Andrianos Michail

**Abstract**: We extend BeamAttack, an adversarial attack algorithm designed to evaluate the robustness of text classification systems through word-level modifications guided by beam search. Our extensions include support for word deletions and the option to skip substitutions, enabling the discovery of minimal modifications that alter model predictions. We also integrate LIME to better prioritize word replacements. Evaluated across multiple datasets and victim models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA framework, our approach achieves over a 99\% attack success rate while preserving the semantic and lexical similarity of the original texts. Through both quantitative and qualitative analysis, we highlight BeamAttack's effectiveness and its limitations. Our implementation is available at https://github.com/LucK1Y/BeamAttack

摘要: 我们扩展了BeamAttack，这是一种对抗攻击算法，旨在通过束搜索指导的词级修改来评估文本分类系统的稳健性。我们的扩展包括对字词删除和跳过替换的选项，从而能够发现改变模型预测的最小修改。我们还集成了LIME，以更好地优先考虑单词替换。在BODEGA框架内对多个数据集和受害者模型（BiLSTM、BERT和对抗训练的RoBERTa）进行评估，我们的方法实现了超过99%的攻击成功率，同时保留了原始文本的语义和词汇相似性。通过定量和定性分析，我们强调了BeamAttack的有效性及其局限性。我们的实施可在https://github.com/LucK1Y/BeamAttack上获取



## **25. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection**

PBCAT：针对对象检测物理上可实现的攻击的基于补丁的复合对抗训练 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23581v1) [paper-pdf](http://arxiv.org/pdf/2506.23581v1)

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.

摘要: 对象检测在许多安全敏感应用程序中发挥着至关重要的作用。然而，最近的几项研究表明，对象检测器很容易被物理上可实现的攻击所愚弄，例如对抗补丁和最近的对抗纹理，这些攻击构成了现实而紧迫的威胁。对抗训练（AT）被认为是对抗攻击的最有效防御。虽然AT在分类模型的$l_\infty$攻击设置中得到了广泛研究，但针对对象检测器物理上可实现的攻击的AT的探索有限。早期的尝试只是为了防御对抗补丁，这使得AT能够对抗更广泛的物理可实现的攻击，但尚未得到充分的探索。在这项工作中，我们考虑使用统一的AT方法来防御各种物理上可实现的攻击。我们提出了PBCAT，这是一种新型的基于补丁的复合对抗训练策略。PBCAT通过结合小区域梯度引导的对抗补丁和覆盖整个图像的不可感知的全局对抗扰动来优化模型。通过这些设计，PBCAT不仅有潜力防御对抗补丁，还有潜力防御不可见的物理可实现的攻击，例如对抗纹理。在多个环境中进行的大量实验表明，与最先进的防御方法相比，PBCAT显着提高了针对各种物理可实现攻击的鲁棒性。值得注意的是，在最近的一次对抗性纹理攻击下，它比之前的防御方法提高了29.7%。



## **26. Efficient Resource Allocation under Adversary Attacks: A Decomposition-Based Approach**

敌对攻击下的高效资源分配：基于分解的方法 cs.DS

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23442v1) [paper-pdf](http://arxiv.org/pdf/2506.23442v1)

**Authors**: Mansoor Davoodi, Setareh Maghsudi

**Abstract**: We address the problem of allocating limited resources in a network under persistent yet statistically unknown adversarial attacks. Each node in the network may be degraded, but not fully disabled, depending on its available defensive resources. The objective is twofold: to minimize total system damage and to reduce cumulative resource allocation and transfer costs over time. We model this challenge as a bi-objective optimization problem and propose a decomposition-based solution that integrates chance-constrained programming with network flow optimization. The framework separates the problem into two interrelated subproblems: determining optimal node-level allocations across time slots, and computing efficient inter-node resource transfers. We theoretically prove the convergence of our method to the optimal solution that would be obtained with full statistical knowledge of the adversary. Extensive simulations demonstrate that our method efficiently learns the adversarial patterns and achieves substantial gains in minimizing both damage and operational costs, comparing three benchmark strategies under various parameter settings.

摘要: 我们解决了在持续但统计上未知的对抗性攻击下在网络中分配有限资源的问题。网络中的每个节点可能会降级，但不会完全禁用，具体取决于其可用的防御资源。目标是双重的：最大限度地减少系统的总体损害，并随着时间的推移减少累积资源分配和转移成本。我们将这一挑战建模为双目标优化问题，并提出一种基于分解的解决方案，将机会约束规划与网络流优化集成在一起。该框架将问题分为两个相互关联的子问题：确定跨时段的最佳节点级分配，以及计算高效的节点间资源传输。我们从理论上证明了我们的方法收敛于最优解，该最优解将通过对手的充分统计知识获得。广泛的模拟表明，我们的方法可以有效地学习对抗模式，并在最大限度地减少损害和运营成本方面取得了实质性进展，通过比较不同参数设置下的三种基准策略。



## **27. TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs**

TuCo：衡量微调对LLM个人响应的贡献 cs.CL

ICML 2025

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23423v1) [paper-pdf](http://arxiv.org/pdf/2506.23423v1)

**Authors**: Felipe Nuti, Tim Franzmeyer, João Henriques

**Abstract**: Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa.

摘要: 过去的工作研究了微调对大型语言模型（LLM）在某些任务上整体性能的影响。然而，仍然缺乏一种定量、系统的方法来分析其对单个产出的影响。在这里，我们提出了一种新的方法来衡量微调对个体LLM响应的贡献，假设可以访问原始的预训练模型。我们的方法跟踪模型的中间隐藏状态，与预训练和微调模型的最终输出的简单比较相比，提供了对微调效果的更细粒度的见解。我们引入并从理论上分析将任何微调LLM精确分解为预训练组件和微调组件。从经验上看，我们发现模型行为和性能可以通过在前向传递期间放大或缩小微调组件来引导。受这一发现和理论分析的启发，我们将调整贡献（TuCo）定义为微调分量与预训练分量的幅度之比。我们观察到，针对LLM的三种突出的对抗性攻击以某种程度上减少了TuCo的方式规避了安全措施，并且与失败的情况相比，TuCo在这些攻击成功的提示上始终较低。这表明减弱微调对模型输出的影响在此类攻击的成功中发挥了作用。总之，TuCo能够定量研究微调如何影响模型行为和安全性，反之亦然。



## **28. Enhancing Adversarial Robustness through Multi-Objective Representation Learning**

通过多目标表示学习增强对抗鲁棒性 cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2410.01697v4) [paper-pdf](http://arxiv.org/pdf/2410.01697v4)

**Authors**: Sedjro Salomon Hotegni, Sebastian Peitz

**Abstract**: Deep neural networks (DNNs) are vulnerable to small adversarial perturbations, which are tiny changes to the input data that appear insignificant but cause the model to produce drastically different outputs. Many defense methods require modifying model architectures during evaluation or performing test-time data purification. This not only introduces additional complexity but is often architecture-dependent. We show, however, that robust feature learning during training can significantly enhance DNN robustness. We propose MOREL, a multi-objective approach that aligns natural and adversarial features using cosine similarity and multi-positive contrastive losses to encourage similar features for same-class inputs. Extensive experiments demonstrate that MOREL significantly improves robustness against both white-box and black-box attacks. Our code is available at https://github.com/salomonhotegni/MOREL

摘要: 深度神经网络（DNN）容易受到微小的对抗性扰动的影响，这些扰动是对输入数据的微小变化，这些变化看起来微不足道，但会导致模型产生截然不同的输出。许多防御方法需要在评估或执行测试时数据净化期间修改模型架构。这不仅带来了额外的复杂性，而且通常取决于体系结构。然而，我们表明，训练期间的稳健特征学习可以显着增强DNN稳健性。我们提出了MOREL，这是一种多目标方法，它使用余弦相似性和多正对比损失来对齐自然和对抗特征，以鼓励同类输入的相似特征。大量的实验表明，MOREL显着提高了对白盒和黑盒攻击的鲁棒性。我们的代码可在https://github.com/salomonhotegni/MOREL上获取



## **29. Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning**

联邦学习中通过后门攻击解除对抗鲁棒性 cs.LG

15 pages, 8 main pages of text, 13 figures, 5 tables. Made for a  Neurips workshop on backdoor attacks - extended version

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2310.11594v3) [paper-pdf](http://arxiv.org/pdf/2310.11594v3)

**Authors**: Taejin Kim, Jiarui Li, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: The delicate equilibrium between user privacy and the ability to unleash the potential of distributed data is an important concern. Federated learning, which enables the training of collaborative models without sharing of data, has emerged as a privacy-centric solution. This approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data into the training process, as well as evasion attacks that aim to induce misclassifications at test time. Our research investigates the intersection of adversarial training, a common defense method against evasion attacks, and backdoor attacks within federated learning. We introduce Adversarial Robustness Unhardening (ARU), which is employed by a subset of adversarial clients to intentionally undermine model robustness during federated training, rendering models susceptible to a broader range of evasion attacks. We present extensive experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning and backdoor attacks. Our results show that ARU can substantially undermine adversarial training's ability to harden models against test-time evasion attacks, and that adversaries employing ARU can even evade robust aggregation defenses that often neutralize poisoning or backdoor attacks.

摘要: 用户隐私与释放分布式数据潜力的能力之间的微妙平衡是一个重要问题。联合学习可以在不共享数据的情况下训练协作模型，已成为一种以隐私为中心的解决方案。这种方法带来了安全挑战，特别是恶意实体将损坏的数据注入到训练过程中的中毒和后门攻击，以及旨在在测试时引发错误分类的规避攻击。我们的研究调查了联邦学习中对抗训练、针对规避攻击的常见防御方法和后门攻击的交叉点。我们引入了对抗稳健性解除硬化（ARU），一部分对抗客户端使用它来在联邦训练期间故意破坏模型稳健性，使模型容易受到更广泛的规避攻击。我们进行了广泛的实验，评估ARU对对抗训练以及针对中毒和后门攻击的现有强大聚集防御的影响。我们的结果表明，ARU可以极大地削弱对抗训练强化模型抵御测试时规避攻击的能力，而且使用ARU的对手甚至可以规避通常可以中和中毒或后门攻击的强大聚集防御。



## **30. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2411.16782v3) [paper-pdf](http://arxiv.org/pdf/2411.16782v3)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.

摘要: 对抗性示例通常表现出良好的跨模型可移植性，从而能够在有关其架构和参数的有限信息的情况下对黑匣子模型进行攻击，这在商业黑匣子场景中具有高度威胁性。模型集成是通过攻击多个代理模型来提高对抗性示例可移植性的有效策略。然而，由于之前的研究通常在整体中采用很少的模型，因此扩大模型数量是否可以进一步改善黑匣子攻击仍然是一个悬而未决的问题。受大型基金会模型缩放定律的启发，我们在这项工作中研究了黑匣子对抗攻击的缩放定律。通过理论分析和实证评估，我们得出了明确的缩放定律，即使用更多的代理模型增强了对抗性可转让性。全面的实验验证了标准图像分类器、多样化防御模型和使用各种对抗攻击方法的多模式大型语言模型的主张。具体来说，通过缩放定律，即使是GPT-4 o等专有模型，我们也能实现90%以上的传输攻击成功率。进一步的可视化表明，对抗性扰动的可解释性和语义也存在缩放定律。



## **31. MedLeak: Multimodal Medical Data Leakage in Secure Federated Learning with Crafted Models**

MedLeak：使用精心设计的模型的安全联邦学习中的多模式医疗数据泄露 cs.LG

Accepted by the IEEE/ACM conference on Connected Health:  Applications, Systems and Engineering Technologies 2025 (CHASE'25)

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2407.09972v2) [paper-pdf](http://arxiv.org/pdf/2407.09972v2)

**Authors**: Shanghao Shi, Md Shahedul Haque, Abhijeet Parida, Chaoyu Zhang, Marius George Linguraru, Y. Thomas Hou, Syed Muhammad Anwar, Wenjing Lou

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine learning models while keeping their data local, making it ideal for collaborations among healthcare institutions on sensitive data. However, in this paper, we propose a novel privacy attack called MedLeak, which allows a malicious FL server to recover high-quality site-specific private medical data from the client model updates. MedLeak works by introducing an adversarially crafted model during the FL training process. Honest clients, unaware of the insidious changes in the published models, continue to send back their updates as per the standard FL protocol. Leveraging a novel analytical method, MedLeak can efficiently recover private client data from the aggregated parameter updates, eliminating costly optimization. In addition, the scheme relies solely on the aggregated updates, thus rendering secure aggregation protocols ineffective, as they depend on the randomization of intermediate results for security while leaving the final aggregated results unaltered.   We implement MedLeak on medical image datasets (MedMNIST, COVIDx CXR-4, and Kaggle Brain Tumor MRI), as well as a medical text dataset (MedAbstract). The results demonstrate that our attack achieves high recovery rates and strong quantitative scores on both image and text datasets. We also thoroughly evaluate MedLeak across different attack parameters, providing insights into key factors that influence attack performance and potential defenses. Furthermore, we demonstrate that the recovered data can support downstream tasks such as disease classification with minimal performance loss. Our findings validate the need for enhanced privacy measures in FL systems, particularly for safeguarding sensitive medical data against powerful model inversion attacks.

摘要: 联合学习（FL）允许参与者协作训练机器学习模型，同时将其数据保持在本地，因此非常适合医疗机构之间就敏感数据进行合作。然而，在本文中，我们提出了一种名为MedLeak的新型隐私攻击，它允许恶意FL服务器从客户端模型更新中恢复高质量的特定站点私人医疗数据。MedLeak的工作原理是在FL培训过程中引入一个对抗性制作的模型。诚实的客户没有意识到已发布模型中的潜在变化，会继续根据标准FL协议发送更新。利用新颖的分析方法，MedLeak可以从聚合参数更新中有效地恢复私人客户数据，从而消除昂贵的优化。此外，该方案仅依赖于聚合更新，从而导致安全聚合协议无效，因为它们依赖于中间结果的随机化以实现安全，而最终的聚合结果保持不变。   我们在医学图像数据集（MedMNIST、COVIDx CXR-4和Kaggle脑肿瘤MRI）以及医学文本数据集（MedAbstract）上实施MedLeak。结果表明，我们的攻击在图像和文本数据集上都实现了高恢复率和强大的量化分数。我们还对不同的攻击参数进行了彻底评估，从而深入了解影响攻击性能和潜在防御的关键因素。此外，我们证明恢复的数据可以以最小的性能损失支持疾病分类等下游任务。我们的研究结果证实了FL系统中增强隐私措施的必要性，特别是为了保护敏感医疗数据免受强大的模型倒置攻击。



## **32. Securing AI Systems: A Guide to Known Attacks and Impacts**

保护人工智能系统：已知攻击和影响指南 cs.CR

34 pages, 16 figures

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23296v1) [paper-pdf](http://arxiv.org/pdf/2506.23296v1)

**Authors**: Naoto Kiribuchi, Kengo Zenitani, Takayuki Semitsu

**Abstract**: Embedded into information systems, artificial intelligence (AI) faces security threats that exploit AI-specific vulnerabilities. This paper provides an accessible overview of adversarial attacks unique to predictive and generative AI systems. We identify eleven major attack types and explicitly link attack techniques to their impacts -- including information leakage, system compromise, and resource exhaustion -- mapped to the confidentiality, integrity, and availability (CIA) security triad. We aim to equip researchers, developers, security practitioners, and policymakers, even those without specialized AI security expertise, with foundational knowledge to recognize AI-specific risks and implement effective defenses, thereby enhancing the overall security posture of AI systems.

摘要: 人工智能（AI）嵌入到信息系统中，面临着利用人工智能特定漏洞的安全威胁。本文提供了预测和生成人工智能系统特有的对抗性攻击的易于理解的概述。我们确定了十一种主要攻击类型，并将攻击技术与其影响明确联系起来（包括信息泄露、系统泄露和资源耗尽），映射到机密性、完整性和可用性（CIA）安全三重位。我们的目标是为研究人员、开发人员、安全从业者和政策制定者（即使是那些没有专业人工智能安全专业知识的人）提供识别人工智能特定风险并实施有效防御的基础知识，从而增强人工智能系统的整体安全态势。



## **33. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

从即时注入到协议漏洞：LLM支持的人工智能代理工作流程中的威胁 cs.CR

29 pages, 15 figures, 6 tables

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23260v1) [paper-pdf](http://arxiv.org/pdf/2506.23260v1)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces have dramatically expanded capabilities for real-time data retrieval, complex computation, and multi-step orchestration. Yet, the explosive proliferation of plugins, connectors, and inter-agent protocols has outpaced discovery mechanisms and security practices, resulting in brittle integrations vulnerable to diverse threats. In this survey, we introduce the first unified, end-to-end threat model for LLM-agent ecosystems, spanning host-to-tool and agent-to-agent communications, formalize adversary capabilities and attacker objectives, and catalog over thirty attack techniques. Specifically, we organized the threat model into four domains: Input Manipulation (e.g., prompt injections, long-context hijacks, multimodal adversarial inputs), Model Compromise (e.g., prompt- and parameter-level backdoors, composite and encrypted multi-backdoors, poisoning strategies), System and Privacy Attacks (e.g., speculative side-channels, membership inference, retrieval poisoning, social-engineering simulations), and Protocol Vulnerabilities (e.g., exploits in Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent Network Protocol (ANP), and Agent-to-Agent (A2A) protocol). For each category, we review representative scenarios, assess real-world feasibility, and evaluate existing defenses. Building on our threat taxonomy, we identify key open challenges and future research directions, such as securing MCP deployments through dynamic trust management and cryptographic provenance tracking; designing and hardening Agentic Web Interfaces; and achieving resilience in multi-agent and federated environments. Our work provides a comprehensive reference to guide the design of robust defense mechanisms and establish best practices for resilient LLM-agent workflows.

摘要: 由具有结构化功能调用接口的大型语言模型（LLM）支持的自主人工智能代理极大地扩展了实时数据检索、复杂计算和多步骤编排的能力。然而，插件、连接器和代理间协议的爆炸性激增已经超过了发现机制和安全实践的速度，导致集成脆弱，容易受到各种威胁的影响。在本调查中，我们为LLM代理生态系统引入了第一个统一的端到端威胁模型，涵盖主机到工具和代理到代理的通信，正式化对手能力和攻击者目标，并对三十多种攻击技术进行了分类。具体来说，我们将威胁模型组织为四个领域：输入操纵（例如，提示注入、长上下文劫持、多模式对抗输入）、模型妥协（例如，提示和参数级后门、复合和加密的多后门、中毒策略）、系统和隐私攻击（例如，推测性侧通道、成员资格推断、检索中毒、社会工程模拟）和协议漏洞（例如，模型上下文协议（HCP）、代理通信协议（ACP）、代理网络协议（ANP）和代理对代理（A2 A）协议中的漏洞利用）。对于每个类别，我们都会审查代表性场景、评估现实世界的可行性并评估现有的防御措施。基于我们的威胁分类法，我们确定了关键的开放挑战和未来的研究方向，例如通过动态信任管理和加密来源跟踪来保护LCP部署;设计和强化统计Web界面;以及在多代理和联邦环境中实现弹性。我们的工作提供了全面的参考，以指导稳健的防御机制的设计并为弹性LLM代理工作流程建立最佳实践。



## **34. Fragile, Robust, and Antifragile: A Perspective from Parameter Responses in Reinforcement Learning Under Stress**

脆弱性、稳健性和反脆弱性：从压力下强化学习中的参数响应的角度来看 cs.LG

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.23036v1) [paper-pdf](http://arxiv.org/pdf/2506.23036v1)

**Authors**: Zain ul Abdeen, Ming Jin

**Abstract**: This paper explores Reinforcement learning (RL) policy robustness by systematically analyzing network parameters under internal and external stresses. Inspired by synaptic plasticity in neuroscience, synaptic filtering introduces internal stress by selectively perturbing parameters, while adversarial attacks apply external stress through modified agent observations. This dual approach enables the classification of parameters as fragile, robust, or antifragile, based on their influence on policy performance in clean and adversarial settings. Parameter scores are defined to quantify these characteristics, and the framework is validated on PPO-trained agents in Mujoco continuous control environments. The results highlight the presence of antifragile parameters that enhance policy performance under stress, demonstrating the potential of targeted filtering techniques to improve RL policy adaptability. These insights provide a foundation for future advancements in the design of robust and antifragile RL systems.

摘要: 本文通过系统分析内部和外部压力下的网络参数来探索强化学习（RL）策略的稳健性。受神经科学中突触可塑性的启发，突触过滤通过选择性地扰乱参数来引入内部压力，而对抗攻击则通过修改的主体观察来施加外部压力。这种双重方法可以根据参数在干净和敌对环境中对政策绩效的影响将参数分类为脆弱、稳健或反脆弱。定义参数分数来量化这些特征，并在Mujoco连续控制环境中的PPA训练代理上验证了该框架。结果强调了反脆弱参数的存在，可以增强压力下的政策性能，证明了有针对性的过滤技术提高RL政策适应性的潜力。这些见解为未来鲁棒和抗脆弱RL系统设计的进步提供了基础。



## **35. Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models**

重温CroPA：视觉语言模型中交叉提示对抗可移植性的再现性研究和增强 cs.CV

Accepted to MLRC 2025

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22982v1) [paper-pdf](http://arxiv.org/pdf/2506.22982v1)

**Authors**: Atharv Mittal, Agam Pandey, Amritanshu Tiwari, Sukrit Jindal, Swadesh Swain

**Abstract**: Large Vision-Language Models (VLMs) have revolutionized computer vision, enabling tasks such as image classification, captioning, and visual question answering. However, they remain highly vulnerable to adversarial attacks, particularly in scenarios where both visual and textual modalities can be manipulated. In this study, we conduct a comprehensive reproducibility study of "An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models" validating the Cross-Prompt Attack (CroPA) and confirming its superior cross-prompt transferability compared to existing baselines. Beyond replication we propose several key improvements: (1) A novel initialization strategy that significantly improves Attack Success Rate (ASR). (2) Investigate cross-image transferability by learning universal perturbations. (3) A novel loss function targeting vision encoder attention mechanisms to improve generalization. Our evaluation across prominent VLMs -- including Flamingo, BLIP-2, and InstructBLIP as well as extended experiments on LLaVA validates the original results and demonstrates that our improvements consistently boost adversarial effectiveness. Our work reinforces the importance of studying adversarial vulnerabilities in VLMs and provides a more robust framework for generating transferable adversarial examples, with significant implications for understanding the security of VLMs in real-world applications.

摘要: 大型视觉语言模型（VLM）彻底改变了计算机视觉，实现了图像分类、字幕和视觉问答等任务。然而，它们仍然非常容易受到对抗攻击，特别是在视觉和文本模式都可以被操纵的场景中。在这项研究中，我们对“一个图像值得1000个谎言：视觉语言模型上的冲突可移植性”进行了全面的重复性研究，验证了交叉提示攻击（CroPA），并确认了与现有基线相比其优越的交叉提示可移植性。除了复制之外，我们还提出了几项关键改进：（1）一种新颖的初始化策略，可以显着提高攻击成功率（ASB）。(2)通过学习普适扰动来研究跨图像的可移植性。(3)一种针对视觉编码器注意力机制的新型损失函数，以提高概括性。我们对著名VLM（包括Flamingo、BLIP-2和INSTBLIP）的评估以及LLaVA的扩展实验验证了原始结果，并证明我们的改进持续提高了对抗有效性。我们的工作强调了研究VLM中对抗性漏洞的重要性，并为生成可转移的对抗性示例提供了一个更强大的框架，这对于理解现实世界应用程序中的VLM的安全性具有重要意义。



## **36. VFEFL: Privacy-Preserving Federated Learning against Malicious Clients via Verifiable Functional Encryption**

VFEFL：通过可验证的功能加密针对恶意客户端保护隐私的联邦学习 cs.CR

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.12846v2) [paper-pdf](http://arxiv.org/pdf/2506.12846v2)

**Authors**: Nina Cai, Jinguang Han, Weizhi Meng

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods.

摘要: 联邦学习是一种有前途的分布式学习范式，可以在不暴露本地客户数据的情况下实现协作模型训练，从而保护数据隐私。然而，它也带来了新的威胁和挑战。模型倒置攻击的发展使本地模型的明文传输变得不安全，而联邦学习的分布式性质使其特别容易受到恶意客户端发起的攻击。为了保护数据隐私并防止恶意客户端攻击，本文提出了一种基于可验证功能加密的隐私保护联邦学习框架，无需非勾结双服务器设置或额外的受信任第三方。具体来说，我们提出了一种新型的去中心化可验证功能加密（DVFE）方案，该方案能够验证多维密文上的特定关系。从定义、安全模型和安全证明方面对该方案进行了正式处理。此外，基于提出的DVFE方案，我们设计了一个保护隐私的联邦学习框架VFEFL，该框架结合了一种新颖的鲁棒聚合规则来检测恶意客户端，从而能够在对抗环境下有效训练高准确度模型。最后，我们对所提出的方案进行了形式分析和实证评估。结果表明，我们的方法实现了预期的隐私保护、稳健性、可验证性和保真度，同时消除了对现有方法所需的非勾结双服务器设置或受信任第三方的依赖。



## **37. Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate**

通过剩余注意力门的文本到图像扩散模型的概念精确擦除器 cs.CV

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22806v1) [paper-pdf](http://arxiv.org/pdf/2506.22806v1)

**Authors**: Byung Hyun Lee, Sungjin Lim, Seunggyu Lee, Dong Un Kang, Se Young Chun

**Abstract**: Remarkable progress in text-to-image diffusion models has brought a major concern about potentially generating images on inappropriate or trademarked concepts. Concept erasing has been investigated with the goals of deleting target concepts in diffusion models while preserving other concepts with minimal distortion. To achieve these goals, recent concept erasing methods usually fine-tune the cross-attention layers of diffusion models. In this work, we first show that merely updating the cross-attention layers in diffusion models, which is mathematically equivalent to adding \emph{linear} modules to weights, may not be able to preserve diverse remaining concepts. Then, we propose a novel framework, dubbed Concept Pinpoint Eraser (CPE), by adding \emph{nonlinear} Residual Attention Gates (ResAGs) that selectively erase (or cut) target concepts while safeguarding remaining concepts from broad distributions by employing an attention anchoring loss to prevent the forgetting. Moreover, we adversarially train CPE with ResAG and learnable text embeddings in an iterative manner to maximize erasing performance and enhance robustness against adversarial attacks. Extensive experiments on the erasure of celebrities, artistic styles, and explicit contents demonstrated that the proposed CPE outperforms prior arts by keeping diverse remaining concepts while deleting the target concepts with robustness against attack prompts. Code is available at https://github.com/Hyun1A/CPE

摘要: 文本到图像扩散模型的显着进展引发了人们对可能基于不恰当或商标概念生成图像的重大担忧。人们对概念擦除进行了研究，目标是删除扩散模型中的目标概念，同时以最小的失真保留其他概念。为了实现这些目标，最近的概念删除方法通常会微调扩散模型的交叉注意层。在这项工作中，我们首先表明，仅仅更新扩散模型中的交叉注意层（在数学上相当于将\{线性}模块添加到权重中）可能无法保留多样化的剩余概念。然后，我们提出了一个名为概念精确擦除器（CPD）的新颖框架，通过添加\{非线性}剩余注意力门（ResAG），该门选择性地擦除（或切割）目标概念，同时通过采用注意力锚定损失来保护剩余概念免受广泛分布的影响来防止遗忘。此外，我们以迭代的方式使用ResAG和可学习文本嵌入来对抗性训练CPD，以最大限度地提高擦除性能并增强对抗性攻击的鲁棒性。关于删除名人、艺术风格和明确内容的广泛实验表明，拟议的CPD通过保留多样的剩余概念，同时删除目标概念来对攻击提示具有鲁棒性，从而优于现有技术。代码可访问https://github.com/Hyun1A/CPE



## **38. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2407.01461v3) [paper-pdf](http://arxiv.org/pdf/2407.01461v3)

**Authors**: Xiaohua Wang, Zisu Huang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Qi Qian, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型（LLM）生成诚实、无害且有帮助的响应的能力严重依赖于用户提示的质量。然而，这些提示往往简短且模糊，从而严重限制了法学硕士的全部潜力。此外，对手可能会精心设计和操纵有害提示来越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLM的能力，同时保持针对有害越狱输入的强大鲁棒性，本研究提出了一种可转移且可插入的框架，该框架在用户提示被输入LLM之前对其进行完善。该策略提高了查询的质量，使LLM能够生成更真实、良性和有用的响应。具体来说，使用专门设计的强化学习方法引入并训练轻量级查询细化模型，该方法结合了多个目标以增强LLM的特定能力。大量实验表明，细化模型不仅提高了响应的质量，而且增强了响应对越狱攻击的鲁棒性。代码可访问：https://github.com/Huangzisu/query-refinement。



## **39. Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation**

更小=更弱？量化LLM在代码生成中的基准测试鲁棒性 cs.SE

13 pages, 6 figures

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22776v1) [paper-pdf](http://arxiv.org/pdf/2506.22776v1)

**Authors**: Sen Fang, Weiyuan Ding, Antonio Mastropaolo, Bowen Xu

**Abstract**: Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely unexplored.In this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies.

摘要: 量化已成为压缩大型语言模型（LLM）、减少内存需求并加速推理的主流方法，无需修改架构。虽然现有的研究主要集中在评估量化LLM与原始同类相比的有效性，但对稳健性的影响在很大程度上尚未探索。在本文中，我们首次系统地研究量化如何影响LLM在代码生成任务中的稳健性。通过对四个著名的LLM家族（LLaMA、DeepSeek、CodeGen和StarCoder）进行广泛实验，参数范围从350 M到33 B，我们从双重角度评估稳健性：对输入提示的对抗攻击和模型架构的噪音扰动。我们的研究结果挑战了传统智慧，证明量化LLM通常表现出比全精度同行更出色的鲁棒性，我们的对抗实验中分别有51.59%和42.86%表现出量化LLM更好的弹性。同样，我们的噪音扰动实验也证实，定量后的LLM通常可以承受更高水平的体重扰动。这些结果表明，量化不仅降低了计算要求，而且实际上可以增强LLM在代码生成任务中的可靠性，为开发更稳健、更高效的LLM部署策略提供有价值的见解。



## **40. Kill Two Birds with One Stone! Trajectory enabled Unified Online Detection of Adversarial Examples and Backdoor Attacks**

一举两得！轨迹支持对抗性示例和后门攻击的统一在线检测 cs.CR

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22722v1) [paper-pdf](http://arxiv.org/pdf/2506.22722v1)

**Authors**: Anmin Fu, Fanyu Meng, Huaibing Peng, Hua Ma, Zhi Zhang, Yifeng Zheng, Willy Susilo, Yansong Gao

**Abstract**: The proposed UniGuard is the first unified online detection framework capable of simultaneously addressing adversarial examples and backdoor attacks. UniGuard builds upon two key insights: first, both AE and backdoor attacks have to compromise the inference phase, making it possible to tackle them simultaneously during run-time via online detection. Second, an adversarial input, whether a perturbed sample in AE attacks or a trigger-carrying sample in backdoor attacks, exhibits distinctive trajectory signatures from a benign sample as it propagates through the layers of a DL model in forward inference. The propagation trajectory of the adversarial sample must deviate from that of its benign counterpart; otherwise, the adversarial objective cannot be fulfilled. Detecting these trajectory signatures is inherently challenging due to their subtlety; UniGuard overcomes this by treating the propagation trajectory as a time-series signal, leveraging LSTM and spectrum transformation to amplify differences between adversarial and benign trajectories that are subtle in the time domain. UniGuard exceptional efficiency and effectiveness have been extensively validated across various modalities (image, text, and audio) and tasks (classification and regression), ranging from diverse model architectures against a wide range of AE attacks and backdoor attacks, including challenging partial backdoors and dynamic triggers. When compared to SOTA methods, including ContraNet (NDSS 22) specific for AE detection and TED (IEEE SP 24) specific for backdoor detection, UniGuard consistently demonstrates superior performance, even when matched against each method's strengths in addressing their respective threats-each SOTA fails to parts of attack strategies while UniGuard succeeds for all.

摘要: 拟议的UniGuard是第一个能够同时解决对抗性示例和后门攻击的统一在线检测框架。UniGuard基于两个关键见解：首先，AE和后门攻击都必须损害推理阶段，以便可以通过在线检测在运行时同时解决它们。其次，对抗性输入，无论是AE攻击中的受干扰样本还是后门攻击中的携带样本，在良性样本在前向推理中传播时，都会表现出良性样本的独特轨迹特征。对抗性样本的传播轨迹必须偏离其良性样本的传播轨迹;否则，对抗性目标就无法实现。由于这些轨迹特征的微妙性，检测这些轨迹特征本质上具有挑战性; UniGuard通过将传播轨迹视为时间序列信号，利用LSTM和频谱变换来放大在时间域中微妙的对抗轨迹和良性轨迹之间的差异来克服了这一点。UniGuard卓越的效率和有效性已在各种形式（图像、文本和音频）和任务（分类和回归）中得到了广泛验证，范围从针对广泛的AE攻击和后门攻击的不同模型架构，包括具有挑战性的部分后门和动态触发器。与SOTA方法（包括专用于AE检测的ContrNet（NDSS 22）和专用于后门检测的TED（IEEE SP 24））相比，UniGuard始终表现出卓越的性能，即使与每种方法在解决各自威胁方面的优势相匹配-每个SOTA都失败了部分攻击策略，而UniGuard则成功了所有人。



## **41. VERA: Variational Inference Framework for Jailbreaking Large Language Models**

VERA：越狱大型语言模型的变分推理框架 cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22666v1) [paper-pdf](http://arxiv.org/pdf/2506.22666v1)

**Authors**: Anamika Lochab, Lu Yan, Patrick Pynadath, Xiangyu Zhang, Ruqi Zhang

**Abstract**: The rise of API-only access to state-of-the-art LLMs highlights the need for effective black-box jailbreak methods to identify model vulnerabilities in real-world settings. Without a principled objective for gradient-based optimization, most existing approaches rely on genetic algorithms, which are limited by their initialization and dependence on manually curated prompt pools. Furthermore, these methods require individual optimization for each prompt, failing to provide a comprehensive characterization of model vulnerabilities. To address this gap, we introduce VERA: Variational infErence fRamework for jAilbreaking. VERA casts black-box jailbreak prompting as a variational inference problem, training a small attacker LLM to approximate the target LLM's posterior over adversarial prompts. Once trained, the attacker can generate diverse, fluent jailbreak prompts for a target query without re-optimization. Experimental results show that VERA achieves strong performance across a range of target LLMs, highlighting the value of probabilistic inference for adversarial prompt generation.

摘要: 仅限API访问最先进的LLM的兴起凸显了有效的黑匣子越狱方法来识别现实世界环境中的模型漏洞的必要性。由于没有基于梯度的优化的原则目标，大多数现有方法都依赖于遗传算法，而遗传算法受到初始化和对手动策划提示池的依赖的限制。此外，这些方法需要对每个提示进行单独优化，无法提供模型漏洞的全面描述。为了解决这个差距，我们引入了VERA：变分影响Erence fRamework for jAilbreaking。VERA将黑匣子越狱提示视为变分推理问题，训练小型攻击者LLM在对抗性提示上逼近目标LLM的后验。经过训练后，攻击者可以为目标查询生成多样化、流畅的越狱提示，而无需重新优化。实验结果表明，VERA在一系列目标LLM中实现了强劲的性能，凸显了概率推理对对抗提示生成的价值。



## **42. MetaCipher: A General and Extensible Reinforcement Learning Framework for Obfuscation-Based Jailbreak Attacks on Black-Box LLMs**

MetaCipher：一个通用且可扩展的强化学习框架，用于对黑匣子LLM进行基于模糊的越狱攻击 cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22557v1) [paper-pdf](http://arxiv.org/pdf/2506.22557v1)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: The growing capabilities of large language models (LLMs) have exposed them to increasingly sophisticated jailbreak attacks. Among these, obfuscation-based attacks -- which encrypt malicious content to evade detection -- remain highly effective. By leveraging the reasoning ability of advanced LLMs to interpret encrypted prompts, such attacks circumvent conventional defenses that rely on keyword detection or context filtering. These methods are very difficult to defend against, as existing safety mechanisms are not designed to interpret or decode ciphered content. In this work, we propose \textbf{MetaCipher}, a novel obfuscation-based jailbreak framework, along with a reinforcement learning-based dynamic cipher selection mechanism that adaptively chooses optimal encryption strategies from a cipher pool. This approach enhances jailbreak effectiveness and generalizability across diverse task types, victim LLMs, and safety guardrails. Our framework is modular and extensible by design, supporting arbitrary cipher families and accommodating evolving adversarial strategies. We complement our method with a large-scale empirical analysis of cipher performance across multiple victim LLMs. Within as few as 10 queries, MetaCipher achieves over 92\% attack success rate (ASR) on most recent standard malicious prompt benchmarks against state-of-the-art non-reasoning LLMs, and over 74\% ASR against reasoning-capable LLMs, outperforming all existing obfuscation-based jailbreak methods. These results highlight the long-term robustness and adaptability of our approach, making it more resilient than prior methods in the face of advancing safety measures.

摘要: 大型语言模型（LLM）不断增长的能力使它们面临越来越复杂的越狱攻击。其中，基于模糊的攻击（对恶意内容进行加密以逃避检测）仍然非常有效。通过利用高级LLM的推理能力来解释加密提示，此类攻击绕过了依赖关键字检测或上下文过滤的传统防御措施。这些方法非常难以防御，因为现有的安全机制不是为了解释或解码加密内容而设计的。在这项工作中，我们提出了\textBF{MetaCipher}，这是一种新型的基于模糊的越狱框架，以及一种基于强化学习的动态密码选择机制，该机制从密码池中自适应地选择最佳加密策略。这种方法增强了不同任务类型、受害者LLM和安全护栏的越狱有效性和普遍性。我们的框架是模块化的，可通过设计扩展，支持任意密码族并适应不断发展的对抗策略。我们通过对多个受害LLM的密码性能进行大规模实证分析来补充我们的方法。在短短10个查询内，MetaCipher针对最新的非推理LLM在最新标准恶意提示基准上就达到了超过92%的攻击成功率（ASB），针对具有推理能力的LLM达到了超过74%的攻击成功率，优于所有现有的基于模糊的越狱方法。这些结果凸显了我们方法的长期稳健性和适应性，使其在面对先进的安全措施时比以前的方法更具弹性。



## **43. ARMOR: Robust Reinforcement Learning-based Control for UAVs under Physical Attacks**

ARMOR：物理攻击下无人机的鲁棒强化学习控制 cs.LG

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22423v1) [paper-pdf](http://arxiv.org/pdf/2506.22423v1)

**Authors**: Pritam Dash, Ethan Chan, Nathan P. Lawrence, Karthik Pattabiraman

**Abstract**: Unmanned Aerial Vehicles (UAVs) depend on onboard sensors for perception, navigation, and control. However, these sensors are susceptible to physical attacks, such as GPS spoofing, that can corrupt state estimates and lead to unsafe behavior. While reinforcement learning (RL) offers adaptive control capabilities, existing safe RL methods are ineffective against such attacks. We present ARMOR (Adaptive Robust Manipulation-Optimized State Representations), an attack-resilient, model-free RL controller that enables robust UAV operation under adversarial sensor manipulation. Instead of relying on raw sensor observations, ARMOR learns a robust latent representation of the UAV's physical state via a two-stage training framework. In the first stage, a teacher encoder, trained with privileged attack information, generates attack-aware latent states for RL policy training. In the second stage, a student encoder is trained via supervised learning to approximate the teacher's latent states using only historical sensor data, enabling real-world deployment without privileged information. Our experiments show that ARMOR outperforms conventional methods, ensuring UAV safety. Additionally, ARMOR improves generalization to unseen attacks and reduces training cost by eliminating the need for iterative adversarial training.

摘要: 无人机（UAV）依赖机载传感器来进行感知、导航和控制。然而，这些传感器很容易受到物理攻击，例如GPS欺骗，这可能会破坏状态估计并导致不安全行为。虽然强化学习（RL）提供了自适应控制能力，但现有的安全强化学习方法对此类攻击无效。我们提出了ARMOR（自适应鲁棒操纵优化状态表示），这是一种具有攻击弹性、无模型的RL控制器，可在对抗性传感器操纵下实现鲁棒的无人机操作。ARMOR不是依赖原始传感器观察，而是通过两阶段训练框架学习无人机物理状态的稳健潜在表示。在第一阶段，用特权攻击信息训练的教师编码器生成攻击感知潜在状态用于RL策略训练。在第二阶段，通过监督学习训练学生编码器，仅使用历史传感器数据来逼近教师的潜在状态，从而实现在没有特权信息的情况下的现实世界部署。我们的实验表明，ARMOR优于传统方法，确保了无人机的安全。此外，ARMOR提高了对不可见攻击的概括性，并通过消除迭代对抗训练的需要来降低训练成本。



## **44. Secure Video Quality Assessment Resisting Adversarial Attacks**

安全的视频质量评估抵抗对抗攻击 cs.CV

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2410.06866v2) [paper-pdf](http://arxiv.org/pdf/2410.06866v2)

**Authors**: Ao-Xiang Zhang, Yuan-Gen Wang, Yu Ran, Weixuan Tang, Qingxiao Guan, Chunsheng Yang

**Abstract**: The exponential surge in video traffic has intensified the imperative for Video Quality Assessment (VQA). Leveraging cutting-edge architectures, current VQA models have achieved human-comparable accuracy. However, recent studies have revealed the vulnerability of existing VQA models against adversarial attacks. To establish a reliable and practical assessment system, a secure VQA model capable of resisting such malicious attacks is urgently demanded. Unfortunately, no attempt has been made to explore this issue. This paper first attempts to investigate general adversarial defense principles, aiming at endowing existing VQA models with security. Specifically, we first introduce random spatial grid sampling on the video frame for intra-frame defense. Then, we design pixel-wise randomization through a guardian map, globally neutralizing adversarial perturbations. Meanwhile, we extract temporal information from the video sequence as compensation for inter-frame defense. Building upon these principles, we present a novel VQA framework from the security-oriented perspective, termed SecureVQA. Extensive experiments indicate that SecureVQA sets a new benchmark in security while achieving competitive VQA performance compared with state-of-the-art models. Ablation studies delve deeper into analyzing the principles of SecureVQA, demonstrating their generalization and contributions to the security of leading VQA models.

摘要: 视频流量的指数级激增加剧了视频质量评估（VQA）的紧迫性。利用尖端架构，当前的VQA模型已实现人类可比的准确性。然而，最近的研究揭示了现有VQA模型抵御对抗攻击的脆弱性。为了建立可靠、实用的评估系统，迫切需要一种能够抵抗此类恶意攻击的安全VQA模型。不幸的是，尚未尝试探讨这个问题。本文首先尝试研究一般的对抗性防御原则，旨在赋予现有的VQA模型安全性。具体来说，我们首先在视频帧上引入随机空间网格采样以进行帧内防御。然后，我们通过守护者地图设计像素级随机化，在全球范围内中和对抗性扰动。同时，我们从视频序列中提取时间信息作为帧间防御的补偿。在这些原则的基础上，我们从面向安全的角度提出了一种新型VQA框架，称为SecureVQA。大量实验表明，SecureVQA在安全方面树立了新的基准，同时与最先进的模型相比实现了有竞争力的VQA性能。消融研究更深入地分析SecureVQA的原则，展示其普遍性和对领先VQA模型安全性的贡献。



## **45. A Self-scaled Approximate $\ell_0$ Regularization Robust Model for Outlier Detection**

用于离群点检测的自缩放近似$\ell_0$正规化鲁棒模型 eess.SP

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22277v1) [paper-pdf](http://arxiv.org/pdf/2506.22277v1)

**Authors**: Pengyang Song, Jue Wang

**Abstract**: Robust regression models in the presence of outliers have significant practical relevance in areas such as signal processing, financial econometrics, and energy management. Many existing robust regression methods, either grounded in statistical theory or sparse signal recovery, typically rely on the explicit or implicit assumption of outlier sparsity to filter anomalies and recover the underlying signal or data. However, these methods often suffer from limited robustness or high computational complexity, rendering them inefficient for large-scale problems. In this work, we propose a novel robust regression model based on a Self-scaled Approximate l0 Regularization Model (SARM) scheme. By introducing a self-scaling mechanism into the regularization term, the proposed model mitigates the negative impact of uneven or excessively large outlier magnitudes on robustness. We also develop an alternating minimization algorithm grounded in Proximal Operators and Block Coordinate Descent. We rigorously prove the algorithm convergence. Empirical comparisons with several state-of-the-art robust regression methods demonstrate that SARM not only achieves superior robustness but also significantly improves computational efficiency. Motivated by both the theoretical error bound and empirical observations, we further design a Two-Stage SARM (TSSARM) framework, which better utilizes sample information when the singular values of the design matrix are widely spread, thereby enhancing robustness under certain conditions. Finally, we validate our approach on a real-world load forecasting task. The experimental results show that our method substantially enhances the robustness of load forecasting against adversarial data attacks, which is increasingly critical in the era of heightened data security concerns.

摘要: 存在异常值的稳健回归模型在信号处理、金融计量经济学和能源管理等领域具有重要的实际意义。许多现有的稳健回归方法，无论是基于统计理论还是基于稀疏信号恢复，通常依赖于离群点稀疏性的显式或隐式假设来过滤异常并恢复基础信号或数据。然而，这些方法往往鲁棒性有限或计算复杂性高，导致它们对于大规模问题效率低下。在这项工作中，我们提出了一种基于自缩放近似10正规化模型（SARM）方案的新型鲁棒回归模型。通过在正则化项中引入自缩放机制，该模型减轻了不均匀或过大的离群值幅度对鲁棒性的负面影响。我们还开发了一个交替最小化算法接地在邻近算子和块坐标下降。我们严格证明了算法的收敛性。与几种最先进的鲁棒回归方法的实证比较表明，SARM不仅实现了优越的鲁棒性，而且显着提高了计算效率。在理论误差界和经验观察的激励下，我们进一步设计了一个两阶段SARM（TSARM）框架，当设计矩阵的奇异值广泛传播时，该框架可以更好地利用样本信息，从而增强某些条件下的鲁棒性。最后，我们在现实世界的负荷预测任务中验证了我们的方法。实验结果表明，我们的方法大大增强了负载预测针对对抗性数据攻击的鲁棒性，这在数据安全问题加剧的时代变得越来越重要。



## **46. Enhancing Object Detection Robustness: Detecting and Restoring Confidence in the Presence of Adversarial Patch Attacks**

增强对象检测稳健性：在存在对抗性补丁攻击时检测和恢复信心 cs.CV

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2403.12988v2) [paper-pdf](http://arxiv.org/pdf/2403.12988v2)

**Authors**: Roie Kazoom, Raz Birman, Ofer Hadar

**Abstract**: The widespread adoption of computer vision systems has underscored their susceptibility to adversarial attacks, particularly adversarial patch attacks on object detectors. This study evaluates defense mechanisms for the YOLOv5 model against such attacks. Optimized adversarial patches were generated and placed in sensitive image regions, by applying EigenCAM and grid search to determine optimal placement. We tested several defenses, including Segment and Complete (SAC), Inpainting, and Latent Diffusion Models. Our pipeline comprises three main stages: patch application, object detection, and defense analysis. Results indicate that adversarial patches reduce average detection confidence by 22.06\%. Defenses restored confidence levels by 3.45\% (SAC), 5.05\% (Inpainting), and significantly improved them by 26.61\%, which even exceeds the original accuracy levels, when using the Latent Diffusion Model, highlighting its superior effectiveness in mitigating the effects of adversarial patches.

摘要: 计算机视觉系统的广泛采用强调了它们对对抗性攻击的敏感性，特别是对对象检测器的对抗性补丁攻击。本研究评估了针对此类攻击的YOLOv5模型的防御机制。通过应用EigenCAM和网格搜索来确定最佳位置，生成优化的对抗补丁并将其放置在敏感图像区域中。我们测试了几种防御方法，包括分段和完全（SAC），修复和潜在扩散模型。我们的管道包括三个主要阶段：补丁应用、对象检测和防御分析。结果表明，对抗补丁使平均检测置信度降低了22.06%。使用潜伏扩散模型时，防御使信心水平恢复了3.45%（SAC）、5.05%（修补），并显着提高了26.61%，甚至超过了最初的准确性水平，凸显了其在减轻对抗性斑块影响方面的卓越有效性。



## **47. Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses**

推进越狱策略：利用LLM漏洞和扩展现代防御的混合方法 cs.CL

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21972v1) [paper-pdf](http://arxiv.org/pdf/2506.21972v1)

**Authors**: Mohamed Ahmed, Mohamed Abdelmouty, Mingyu Kim, Gunvanth Kandula, Alex Park, James C. Davis

**Abstract**: The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries.

摘要: 预训练语言模型（PTLM）和大型语言模型（LLM）的进步导致它们在不同的应用程序中广泛采用。尽管取得了成功，但这些模型仍然容易受到利用其固有弱点绕过安全措施的攻击。两种主要的推理阶段威胁是代币级和预算级越狱。令牌级攻击嵌入对抗序列，这些序列可以很好地传输到GPT等黑匣子模型，但留下可检测的模式并依赖于基于梯度的令牌优化，而预算级攻击使用语义结构化的输入来引发有害响应，但依赖于可能不可靠的迭代反馈。为了解决这些方法的互补局限性，我们提出了两种混合方法，集成代币和预算级技术，以增强不同PTLM之间的越狱有效性。GCG + PAIR和新探索的GCG + WordGame混合体在多个Vicuna和Lama模型中进行了评估。GCG + PAIR在无防御模型上始终提高了其组成技术的攻击成功率;例如，在Lama-3上，其攻击成功率（ASB）达到91.6%，比PAIR的58.4%基线大幅提高。与此同时，GCG + WordGame与WordGame的原始表现相媲美，即使在Mistral-Sorry-Bench等更严格的评估者下，也保持了超过80%的高ASB。至关重要的是，这两种混合体都保留了可转移性，并可靠地突破了Gradient Cuff和JB Shield等先进防御，从而完全阻止了单一模式攻击。这些发现暴露了当前安全堆栈中以前未报告的漏洞，强调了原始成功和防御稳健性之间的权衡，并强调了针对适应性对手的全面保障措施的必要性。



## **48. Releasing Inequality Phenomenon in $\ell_{\infty}$-norm Adversarial Training via Input Gradient Distillation**

通过输入梯度蒸馏释放$\ell_{\infty}$-norm对抗训练中的不平等现象 cs.CV

16 pages. Accepted by IEEE TIFS

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2305.09305v3) [paper-pdf](http://arxiv.org/pdf/2305.09305v3)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie, Jianhuang Lai

**Abstract**: Adversarial training (AT) is considered the most effective defense against adversarial attacks. However, a recent study revealed that \(\ell_{\infty}\)-norm adversarial training (\(\ell_{\infty}\)-AT) will also induce unevenly distributed input gradients, which is called the inequality phenomenon. This phenomenon makes the \(\ell_{\infty}\)-norm adversarially trained model more vulnerable than the standard-trained model when high-attribution or randomly selected pixels are perturbed, enabling robust and practical black-box attacks against \(\ell_{\infty}\)-adversarially trained models. In this paper, we propose a simple yet effective method called Input Gradient Distillation (IGD) to release the inequality phenomenon in $\ell_{\infty}$-AT. IGD distills the standard-trained teacher model's equal decision pattern into the $\ell_{\infty}$-adversarially trained student model by aligning input gradients of the student model and the standard-trained model with the Cosine Similarity. Experiments show that IGD can mitigate the inequality phenomenon and its threats while preserving adversarial robustness. Compared to vanilla $\ell_{\infty}$-AT, IGD reduces error rates against inductive noise, inductive occlusion, random noise, and noisy images in ImageNet-C by up to 60\%, 16\%, 50\%, and 21\%, respectively. Other than empirical experiments, we also conduct a theoretical analysis to explain why releasing the inequality phenomenon can improve such robustness and discuss why the severity of the inequality phenomenon varies according to the dataset's image resolution. Our code is available at https://github.com/fhdnskfbeuv/Inuput-Gradient-Distillation

摘要: 对抗训练（AT）被认为是对抗攻击最有效的防御。然而，最近的一项研究表明，\（\ell_{\infty}\）-规范对抗训练（\（\ell_{\infty}\）-AT）也会引起不均匀分布的输入梯度，这被称为不平等现象。当高属性或随机选择的像素受到干扰时，这种现象使得\（\ell_{\infty}\）-规范对抗训练模型比标准训练模型更容易受到攻击，从而能够对\（\ell_{\infty}\）-对抗训练模型进行鲁棒且实用的黑匣子攻击。在本文中，我们提出了一种简单而有效的方法，称为输入梯度蒸馏（IGD），以消除$\ell_{\infty}$-AT中的不等式现象。IGD通过将学生模型和标准训练模型的输入梯度与Cosine相似性对齐，将标准训练教师模型的平等决策模式提炼为$\ell_{\infty}$-对抗训练学生模型。实验表明，IGD可以缓解不平等现象及其威胁，同时保持对抗稳健性。与vanilla $\ell_{\infty}$-AT相比，IGD将ImageNet-C中感应性噪音、感应性遮挡、随机噪音和含噪图像的错误率分别降低了高达60%、16%、50%和21%。除了实证实验之外，我们还进行理论分析来解释为什么释放不平等现象可以提高这种稳健性，并讨论为什么不平等现象的严重程度会根据数据集的图像分辨率而变化。我们的代码可在https://github.com/fhdnskfbeuv/Inuput-Gradient-Distillation上获取



## **49. One Video to Steal Them All: 3D-Printing IP Theft through Optical Side-Channels**

一个视频窃取全部内容：通过光学侧通道的3D打印IP盗窃 cs.CR

17 pages [Extended Version]

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21897v1) [paper-pdf](http://arxiv.org/pdf/2506.21897v1)

**Authors**: Twisha Chattopadhyay, Fabricio Ceschin, Marco E. Garza, Dymytriy Zyunkin, Animesh Chhotaray, Aaron P. Stebner, Saman Zonouz, Raheem Beyah

**Abstract**: The 3D printing industry is rapidly growing and increasingly adopted across various sectors including manufacturing, healthcare, and defense. However, the operational setup often involves hazardous environments, necessitating remote monitoring through cameras and other sensors, which opens the door to cyber-based attacks. In this paper, we show that an adversary with access to video recordings of the 3D printing process can reverse engineer the underlying 3D print instructions. Our model tracks the printer nozzle movements during the printing process and maps the corresponding trajectory into G-code instructions. Further, it identifies the correct parameters such as feed rate and extrusion rate, enabling successful intellectual property theft. To validate this, we design an equivalence checker that quantitatively compares two sets of 3D print instructions, evaluating their similarity in producing objects alike in shape, external appearance, and internal structure. Unlike simple distance-based metrics such as normalized mean square error, our equivalence checker is both rotationally and translationally invariant, accounting for shifts in the base position of the reverse engineered instructions caused by different camera positions. Our model achieves an average accuracy of 90.87 percent and generates 30.20 percent fewer instructions compared to existing methods, which often produce faulty or inaccurate prints. Finally, we demonstrate a fully functional counterfeit object generated by reverse engineering 3D print instructions from video.

摘要: 3D打印行业正在迅速发展，并在制造业、医疗保健和国防等各个行业越来越多地采用。然而，操作设置通常涉及危险的环境，需要通过摄像头和其他传感器进行远程监控，这为基于网络的攻击打开了大门。在本文中，我们表明，能够访问3D打印过程视频记录的对手可以对底层3D打印指令进行反向工程。我们的模型在打印过程中跟踪打印机喷嘴的移动，并将相应的轨迹映射到G代码指令中。此外，它还可以识别正确的参数，例如给料率和挤出率，从而实现成功的知识产权盗窃。为了验证这一点，我们设计了一个等效性检查器，该检查器定量比较两组3D打印指令，评估它们在生成形状、外观和内部结构相似的对象时的相似性。与简单的基于距离的指标（例如标准化均方误差）不同，我们的等效检查器在旋转和平移上都是不变的，可以考虑不同摄像机位置引起的反向工程指令基本位置的变化。与经常产生有缺陷或不准确的印刷品的现有方法相比，我们的模型实现了90.87%的平均准确性，生成的指令减少了30.20%。最后，我们演示了通过从视频中进行反向工程3D打印指令生成的功能齐全的伪造对象。



## **50. On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling**

论通过对抗性错误标签毒害文本到图像人工智能模型的可行性 cs.CR

ACM Conference on Computer and Communications Security 2025

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21874v1) [paper-pdf](http://arxiv.org/pdf/2506.21874v1)

**Authors**: Stanley Wu, Ronik Bhaskar, Anna Yoo Jeong Ha, Shawn Shan, Haitao Zheng, Ben Y. Zhao

**Abstract**: Today's text-to-image generative models are trained on millions of images sourced from the Internet, each paired with a detailed caption produced by Vision-Language Models (VLMs). This part of the training pipeline is critical for supplying the models with large volumes of high-quality image-caption pairs during training. However, recent work suggests that VLMs are vulnerable to stealthy adversarial attacks, where adversarial perturbations are added to images to mislead the VLMs into producing incorrect captions.   In this paper, we explore the feasibility of adversarial mislabeling attacks on VLMs as a mechanism to poisoning training pipelines for text-to-image models. Our experiments demonstrate that VLMs are highly vulnerable to adversarial perturbations, allowing attackers to produce benign-looking images that are consistently miscaptioned by the VLM models. This has the effect of injecting strong "dirty-label" poison samples into the training pipeline for text-to-image models, successfully altering their behavior with a small number of poisoned samples. We find that while potential defenses can be effective, they can be targeted and circumvented by adaptive attackers. This suggests a cat-and-mouse game that is likely to reduce the quality of training data and increase the cost of text-to-image model development. Finally, we demonstrate the real-world effectiveness of these attacks, achieving high attack success (over 73%) even in black-box scenarios against commercial VLMs (Google Vertex AI and Microsoft Azure).

摘要: 当今的文本到图像生成模型是在来自互联网的数百万张图像上训练的，每个图像都与视觉语言模型（VLM）生成的详细标题配对。训练管道的这一部分对于在训练期间为模型提供大量高质量图像字幕对至关重要。然而，最近的研究表明，VLM很容易受到隐蔽的对抗攻击，对抗性扰动被添加到图像中以误导VLM产生错误的字幕。   本文中，我们探讨了对VLM的对抗性错误标签攻击作为毒害文本到图像模型训练管道的机制的可行性。我们的实验表明，VLM非常容易受到对抗性扰动的影响，这使得攻击者能够生成看似友善的图像，而这些图像始终被VLM模型字幕错误。这的效果是将强“肮脏标签”毒物样本注入文本到图像模型的训练管道中，用少量毒物样本成功改变它们的行为。我们发现，虽然潜在的防御措施可能有效，但它们可能会被适应性攻击者瞄准和规避。这表明猫鼠游戏可能会降低训练数据的质量并增加文本到图像模型开发的成本。最后，我们展示了这些攻击在现实世界中的有效性，即使在针对商业VLM（Google Vertex AI和Microsoft Azure）的黑匣子场景中，也实现了很高的攻击成功率（超过73%）。



