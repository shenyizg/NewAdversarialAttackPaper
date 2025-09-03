# Latest Adversarial Attack Papers
**update at 2025-09-03 10:21:43**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Targeted Physical Evasion Attacks in the Near-Infrared Domain**

近红外领域有针对性的物理规避攻击 cs.CR

To appear in the Proceedings of the Network and Distributed Systems  Security Symposium (NDSS) 2026

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02042v1) [paper-pdf](http://arxiv.org/pdf/2509.02042v1)

**Authors**: Pascal Zimmer, Simon Lachnit, Alexander Jan Zielinski, Ghassan Karame

**Abstract**: A number of attacks rely on infrared light sources or heat-absorbing material to imperceptibly fool systems into misinterpreting visual input in various image recognition applications. However, almost all existing approaches can only mount untargeted attacks and require heavy optimizations due to the use-case-specific constraints, such as location and shape. In this paper, we propose a novel, stealthy, and cost-effective attack to generate both targeted and untargeted adversarial infrared perturbations. By projecting perturbations from a transparent film onto the target object with an off-the-shelf infrared flashlight, our approach is the first to reliably mount laser-free targeted attacks in the infrared domain. Extensive experiments on traffic signs in the digital and physical domains show that our approach is robust and yields higher attack success rates in various attack scenarios across bright lighting conditions, distances, and angles compared to prior work. Equally important, our attack is highly cost-effective, requiring less than US\$50 and a few tens of seconds for deployment. Finally, we propose a novel segmentation-based detection that thwarts our attack with an F1-score of up to 99%.

摘要: 许多攻击依赖于红外光源或吸热材料，以难以察觉的方式欺骗系统误解各种图像识别应用中的视觉输入。然而，几乎所有现有方法只能发起无目标攻击，并且由于特定用例的限制（例如位置和形状）而需要进行大量优化。在本文中，我们提出了一种新颖的、隐蔽的、具有成本效益的攻击来生成有针对性和无针对性的对抗性红外扰动。通过用现成的红外手电筒将透明薄膜的扰动投射到目标物体上，我们的方法是第一个在红外域中可靠地发动无激光定向攻击的方法。对数字和物理领域交通标志的广泛实验表明，与之前的工作相比，我们的方法稳健，在明亮的照明条件、距离和角度的各种攻击场景中可以产生更高的攻击成功率。同样重要的是，我们的攻击具有极高的成本效益，部署所需时间不到50美元，只需几十秒。最后，我们提出了一种新颖的基于分段的检测，可以以高达99%的F1评分阻止我们的攻击。



## **2. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems**

看不到邪恶：引用多对象跟踪系统中对语言视觉关联的对抗攻击 cs.CV

12 pages, 1 figure, 3 tables

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02028v1) [paper-pdf](http://arxiv.org/pdf/2509.02028v1)

**Authors**: Halima Bouzidi, Haoyu Liu, Mohammad Al Faruque

**Abstract**: Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.

摘要: 图像视觉理解推动了高级感知系统的发展，最引人注目的是参考多对象跟踪（RMOT）的新兴范式。通过利用自然语言查询，RMOT系统可以在基于Transformer的时空推理模块的指导下选择性地跟踪满足给定语义描述的对象。端到端（E2 E）RMOT模型进一步统一了Transformer骨干中的特征提取、时间记忆和空间推理，从而在融合的文本-视觉表示上实现了长距离时空建模。尽管有这些进步，RMOT的可靠性和鲁棒性仍然没有得到充分的研究。在本文中，我们从设计逻辑的角度研究了RMOT系统的安全影响，识别了损害语言视觉引用和跟踪对象匹配组件的对抗漏洞。此外，我们还发现了使用基于FIFO的内存的高级RMOT模型中的一个新漏洞，从而对其时空推理进行有针对性且一致的攻击，从而引入了在历史缓冲区中持续存在的错误。我们提出了VEIL，这是一种新型对抗框架，旨在破坏RMOT模型的统一触发匹配机制。我们表明，精心设计的数字和物理扰动可能会破坏跟踪逻辑的可靠性，导致轨道ID开关和端接。我们使用Refer-KITTI数据集进行全面评估，以验证VEIL的有效性，并证明关键大规模应用对安全感知RMOT设计的迫切需求。



## **3. Adversarial Attacks and Defenses in Multivariate Time-Series Forecasting for Smart and Connected Infrastructures**

智能和互联基础设施多元时间序列预测中的对抗性攻击和防御 cs.LG

18 pages, 34 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2408.14875v2) [paper-pdf](http://arxiv.org/pdf/2408.14875v2)

**Authors**: Pooja Krishan, Rohan Mohapatra, Sanchari Das, Saptarshi Sengupta

**Abstract**: The emergence of deep learning models has revolutionized various industries over the last decade, leading to a surge in connected devices and infrastructures. However, these models can be tricked into making incorrect predictions with high confidence, leading to disastrous failures and security concerns. To this end, we explore the impact of adversarial attacks on multivariate time-series forecasting and investigate methods to counter them. Specifically, we employ untargeted white-box attacks, namely the Fast Gradient Sign Method (FGSM) and the Basic Iterative Method (BIM), to poison the inputs to the training process, effectively misleading the model. We also illustrate the subtle modifications to the inputs after the attack, which makes detecting the attack using the naked eye quite difficult. Having demonstrated the feasibility of these attacks, we develop robust models through adversarial training and model hardening. We are among the first to showcase the transferability of these attacks and defenses by extrapolating our work from the benchmark electricity data to a larger, 10-year real-world data used for predicting the time-to-failure of hard disks. Our experimental results confirm that the attacks and defenses achieve the desired security thresholds, leading to a 72.41% and 94.81% decrease in RMSE for the electricity and hard disk datasets respectively after implementing the adversarial defenses.

摘要: 深度学习模型的出现在过去十年里彻底改变了各个行业，导致互联设备和基础设施的激增。然而，这些模型可能会被欺骗以高置信度做出错误的预测，从而导致灾难性的失败和安全问题。为此，我们探讨了对抗性攻击对多元时间序列预测的影响，并研究应对它们的方法。具体来说，我们使用无针对性的白盒攻击，即快速梯度符号法（FGSM）和基本迭代法（BMI），来毒害训练过程的输入，从而有效地误导模型。我们还说明了攻击后对输入的微妙修改，这使得使用肉眼检测攻击变得相当困难。在证明了这些攻击的可行性后，我们通过对抗训练和模型硬化开发了稳健的模型。我们是最早展示这些攻击和防御可移植性的公司之一，通过将我们的工作从基准电力数据外推到用于预测硬盘故障时间的更大的10年现实世界数据。我们的实验结果证实，攻击和防御都达到了预期的安全阈值，实施对抗性防御后，电力和硬盘数据集的RME分别下降了72.41%和94.81%。



## **4. Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks**

评估机器取消学习针对成员推断攻击的防御潜力 cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2508.16150v2) [paper-pdf](http://arxiv.org/pdf/2508.16150v2)

**Authors**: Aristeidis Sidiropoulos, Christos Chrysanthos Nikolaidis, Theodoros Tsiolakis, Nikolaos Pavlidis, Vasilis Perifanis, Pavlos S. Efraimidis

**Abstract**: Membership Inference Attacks (MIAs) pose a significant privacy risk, as they enable adversaries to determine whether a specific data point was included in the training dataset of a model. While Machine Unlearning is primarily designed as a privacy mechanism to efficiently remove private data from a machine learning model without the need for full retraining, its impact on the susceptibility of models to MIA remains an open question. In this study, we systematically assess the vulnerability of models to MIA after applying state-of-art Machine Unlearning algorithms. Our analysis spans four diverse datasets (two from the image domain and two in tabular format), exploring how different unlearning approaches influence the exposure of models to membership inference. The findings highlight that while Machine Unlearning is not inherently a countermeasure against MIA, the unlearning algorithm and data characteristics can significantly affect a model's vulnerability. This work provides essential insights into the interplay between Machine Unlearning and MIAs, offering guidance for the design of privacy-preserving machine learning systems.

摘要: 会员推断攻击（MIA）构成了重大的隐私风险，因为它们使对手能够确定特定数据点是否包含在模型的训练数据集中。虽然Machine Unlearning主要被设计为一种隐私机制，用于有效地从机器学习模型中删除私人数据，而不需要完全重新培训，但它对模型对MIA敏感性的影响仍然是一个悬而未决的问题。在这项研究中，我们在应用最先进的机器取消学习算法后系统评估了模型对MIA的脆弱性。我们的分析跨越了四个不同的数据集（两个来自图像域，两个以表格形式），探索不同的去学习方法如何影响模型对隶属推理的暴露。研究结果强调，虽然机器取消学习本质上并不是针对MIA的对策，但取消学习算法和数据特征可能会显着影响模型的脆弱性。这项工作为机器非学习和MIA之间的相互作用提供了重要见解，为保护隐私的机器学习系统的设计提供了指导。



## **5. Addressing Key Challenges of Adversarial Attacks and Defenses in the Tabular Domain: A Methodological Framework for Coherence and Consistency**

应对表格领域对抗性攻击和防御的关键挑战：一致性和一致性的方法论框架 cs.LG

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2412.07326v3) [paper-pdf](http://arxiv.org/pdf/2412.07326v3)

**Authors**: Yael Itzhakev, Amit Giloni, Yuval Elovici, Asaf Shabtai

**Abstract**: Machine learning models trained on tabular data are vulnerable to adversarial attacks, even in realistic scenarios where attackers only have access to the model's outputs. Since tabular data contains complex interdependencies among features, it presents a unique challenge for adversarial samples which must maintain coherence and respect these interdependencies to remain indistinguishable from benign data. Moreover, existing attack evaluation metrics-such as the success rate, perturbation magnitude, and query count-fail to account for this challenge. To address those gaps, we propose a technique for perturbing dependent features while preserving sample coherence. In addition, we introduce Class-Specific Anomaly Detection (CSAD), an effective novel anomaly detection approach, along with concrete metrics for assessing the quality of tabular adversarial attacks. CSAD evaluates adversarial samples relative to their predicted class distribution, rather than a broad benign distribution. It ensures that subtle adversarial perturbations, which may appear coherent in other classes, are correctly identified as anomalies. We integrate SHAP explainability techniques to detect inconsistencies in model decision-making, extending CSAD for SHAP-based anomaly detection. Our evaluation incorporates both anomaly detection rates with SHAP-based assessments to provide a more comprehensive measure of adversarial sample quality. We evaluate various attack strategies, examining black-box query-based and transferability-based gradient attacks across four target models. Experiments on benchmark tabular datasets reveal key differences in the attacker's risk and effort and attack quality, offering insights into the strengths, limitations, and trade-offs faced by attackers and defenders. Our findings lay the groundwork for future research on adversarial attacks and defense development in the tabular domain.

摘要: 在表格数据上训练的机器学习模型很容易受到对抗攻击，即使在攻击者只能访问模型输出的现实场景中也是如此。由于表格数据包含特征之间复杂的相互依赖关系，因此它对对抗性样本提出了独特的挑战，这些样本必须保持一致性并尊重这些相互依赖关系，以保持与良性数据之间无法区分。此外，现有的攻击评估指标--例如成功率、扰动幅度和查询计数--无法解决这一挑战。为了解决这些差距，我们提出了一种在保持样本一致性的同时扰动相关特征的技术。此外，我们还引入了类别特定异常检测（CSAD），这是一种有效的新型异常检测方法，以及用于评估表格对抗攻击质量的具体指标。CSAD相对于其预测的类别分布而不是广泛的良性分布来评估敌对样本。它确保细微的对抗性扰动（在其他类别中可能看起来是一致的）被正确识别为异常。我们集成了SHAP可解释性技术来检测模型决策中的不一致性，扩展了CSAD用于基于SHAP的异常检测。我们的评估结合了异常检测率和基于SHAP的评估，以提供对抗性样本质量的更全面的衡量标准。我们评估了各种攻击策略，检查了四个目标模型中基于黑匣子查询和基于可移植性的梯度攻击。对基准表格数据集的实验揭示了攻击者的风险和努力以及攻击质量的关键差异，从而深入了解攻击者和防御者面临的优势、限制和权衡。我们的发现为未来对表格领域的对抗性攻击和防御开发的研究奠定了基础。



## **6. SoK: Cybersecurity Assessment of Humanoid Ecosystem**

SoK：类人生物生态系统的网络安全评估 cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2508.17481v2) [paper-pdf](http://arxiv.org/pdf/2508.17481v2)

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.

摘要: 类人猿正在向医疗保健、工业、国防和服务业的实际部署迈进。虽然通常被认为是网络物理系统（CPS），但它们对传统网络软件栈的依赖（例如，Linux操作系统）、机器人操作系统（LOS）中间件和空中更新通道创建了一个独特的安全配置文件，使它们暴露在传统CPS模型无法完全解决的漏洞中。之前的研究主要检查了特定的威胁，例如LiDART欺骗或对抗性机器学习（ML）。这种狭隘的焦点忽视了针对一个组件的攻击如何对整个机器人的互连系统造成连锁伤害。我们通过知识系统化（SoK）来解决这一差距，该知识采用全面的方法，整合机器人、CPS和网络安全领域的碎片化研究。我们为人形机器人引入了一个七层安全模型，组织了整个人形生态系统中的39种已知攻击和35种防御--从硬件到人机交互。在此安全模型的基础上，我们开发了一个具有风险加权评分的量化39 x35攻击-防御矩阵，并通过蒙特卡洛分析进行验证。我们通过评估三个现实世界的机器人：Pepper、G1 EDU和Digit来演示我们的方法。评分分析显示，各个平台的安全成熟度水平各不相同，评分范围从39.9%到79.5%不等。这项工作引入了一种结构化的、基于证据的评估方法，可以实现系统性的安全评估，支持跨平台基准测试，并指导人形机器人安全投资的优先顺序。



## **7. Privacy-preserving authentication for military 5G networks**

军用5G网络隐私保护认证 cs.CR

To appear in Proc. IEEE Military Commun. Conf. (MILCOM), (Los  Angeles, CA), Oct. 2025

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01470v1) [paper-pdf](http://arxiv.org/pdf/2509.01470v1)

**Authors**: I. D. Lutz, A. M. Hill, M. C. Valenti

**Abstract**: As 5G networks gain traction in defense applications, ensuring the privacy and integrity of the Authentication and Key Agreement (AKA) protocol is critical. While 5G AKA improves upon previous generations by concealing subscriber identities, it remains vulnerable to replay-based synchronization and linkability threats under realistic adversary models. This paper provides a unified analysis of the standardized 5G AKA flow, identifying several vulnerabilities and highlighting how each exploits protocol behavior to compromise user privacy. To address these risks, we present five lightweight mitigation strategies. We demonstrate through prototype implementation and testing that these enhancements strengthen resilience against linkability attacks with minimal computational and signaling overhead. Among the solutions studied, those introducing a UE-generated nonce emerge as the most promising, effectively neutralizing the identified tracking and correlation attacks with negligible additional overhead. Integrating this extension as an optional feature to the standard 5G AKA protocol offers a backward-compatible, low-overhead path toward a more privacy-preserving authentication framework for both commercial and military 5G deployments.

摘要: 随着5G网络在国防应用中获得关注，确保认证和密钥协议（AKA）协议的隐私性和完整性至关重要。虽然5G AKA通过隐藏用户身份对前几代产品进行了改进，但在现实的对手模型下，它仍然容易受到基于回放的同步和链接性威胁。本文对标准化的5G AKA流程进行了统一分析，识别了几个漏洞，并强调了每个漏洞如何利用协议行为来损害用户隐私。为了解决这些风险，我们提出了五种轻量级缓解策略。我们通过原型实现和测试证明，这些增强功能以最小的计算和信号负担增强了抵御链接性攻击的弹性。在所研究的解决方案中，引入UE生成的随机数的解决方案是最有希望的，可以有效地中和已识别的跟踪和相关攻击，而额外费用可以忽略不计。将此扩展集成为标准5G AKA协议的可选功能，为商业和军用5G部署提供了一条向后兼容、低开销的路径，以建立更保护隐私的身份验证框架。



## **8. LLMHoney: A Real-Time SSH Honeypot with Large Language Model-Driven Dynamic Response Generation**

LLMHoney：具有大型语言模型驱动动态响应生成的实时SSH蜜罐 cs.CR

7 Pages

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01463v1) [paper-pdf](http://arxiv.org/pdf/2509.01463v1)

**Authors**: Pranjay Malhotra

**Abstract**: Cybersecurity honeypots are deception tools for engaging attackers and gather intelligence, but traditional low or medium-interaction honeypots often rely on static, pre-scripted interactions that can be easily identified by skilled adversaries. This Report presents LLMHoney, an SSH honeypot that leverages Large Language Models (LLMs) to generate realistic, dynamic command outputs in real time. LLMHoney integrates a dictionary-based virtual file system to handle common commands with low latency while using LLMs for novel inputs, achieving a balance between authenticity and performance. We implemented LLMHoney using open-source LLMs and evaluated it on a testbed with 138 representative Linux commands. We report comprehensive metrics including accuracy (exact-match, Cosine Similarity, Jaro-Winkler Similarity, Levenshtein Similarity and BLEU score), response latency and memory overhead. We evaluate LLMHoney using multiple LLM backends ranging from 0.36B to 3.8B parameters, including both open-source models and a proprietary model(Gemini). Our experiments compare 13 different LLM variants; results show that Gemini-2.0 and moderately-sized models Qwen2.5:1.5B and Phi3:3.8B provide the most reliable and accurate responses, with mean latencies around 3 seconds, whereas smaller models often produce incorrect or out-of-character outputs. We also discuss how LLM integration improves honeypot realism and adaptability compared to traditional honeypots, as well as challenges such as occasional hallucinated outputs and increased resource usage. Our findings demonstrate that LLM-driven honeypots are a promising approach to enhance attacker engagement and collect richer threat intelligence.

摘要: 网络安全蜜罐是用于吸引攻击者和收集情报的欺骗工具，但传统的低或中等交互蜜罐通常依赖于静态的、预先脚本化的交互，这些交互可以被熟练的对手轻松识别。本报告介绍了LLMHoney，这是一个SSH蜜罐，利用大型语言模型（LLM）实时生成真实、动态的命令输出。LLMHoney集成了基于字典的虚拟文件系统，以低延迟处理常见命令，同时使用LLM进行新型输入，实现真实性和性能之间的平衡。我们使用开源LLM实现了LLMHoney，并在具有138个代表性的Linux命令的测试床上对其进行了评估。我们报告了全面的指标，包括准确性（精确匹配、Cosine相似性、Jaro-Winkler相似性、Levenshtein相似性和BLEU评分）、响应延迟和内存负载。我们使用从0.36B到3.8B参数的多个LLM后台来评估LLMHoney，包括开源模型和专有模型（Gemini）。我们的实验比较了13种不同的LLM变体;结果表明，Gemini-2.0和中等大小的模型Qwen 2.5：1.5B和Phi 3：3.8B提供了最可靠和准确的响应，平均延迟时间约为3秒，而较小的模型通常会产生错误或不合字符的输出。我们还讨论了与传统蜜罐相比，LLM集成如何提高蜜罐的真实性和适应性，以及偶尔幻觉输出和资源使用增加等挑战。我们的研究结果表明，LLM驱动的蜜罐是增强攻击者参与度和收集更丰富威胁情报的一种有希望的方法。



## **9. An Automated Attack Investigation Approach Leveraging Threat-Knowledge-Augmented Large Language Models**

利用威胁知识增强大型语言模型的自动攻击调查方法 cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01271v1) [paper-pdf](http://arxiv.org/pdf/2509.01271v1)

**Authors**: Rujie Dai, Peizhuo Lv, Yujiang Gui, Qiujian Lv, Yuanyuan Qiao, Yan Wang, Degang Sun, Weiqing Huang, Yingjiu Li, XiaoFeng Wang

**Abstract**: Advanced Persistent Threats (APTs) are prolonged, stealthy intrusions by skilled adversaries that compromise high-value systems to steal data or disrupt operations. Reconstructing complete attack chains from massive, heterogeneous logs is essential for effective attack investigation, yet existing methods suffer from poor platform generality, limited generalization to evolving tactics, and an inability to produce analyst-ready reports. Large Language Models (LLMs) offer strong semantic understanding and summarization capabilities, but in this domain they struggle to capture the long-range, cross-log dependencies critical for accurate reconstruction.   To solve these problems, we present an LLM-empowered attack investigation framework augmented with a dynamically adaptable Kill-Chain-aligned threat knowledge base. We organizes attack-relevant behaviors into stage-aware knowledge units enriched with semantic annotations, enabling the LLM to iteratively retrieve relevant intelligence, perform causal reasoning, and progressively expand the investigation context. This process reconstructs multi-phase attack scenarios and generates coherent, human-readable investigation reports. Evaluated on 15 attack scenarios spanning single-host and multi-host environments across Windows and Linux (over 4.3M log events, 7.2 GB of data), the system achieves an average True Positive Rate (TPR) of 97.1% and an average False Positive Rate (FPR) of 0.2%, significantly outperforming the SOTA method ATLAS, which achieves an average TPR of 79.2% and an average FPR of 29.1%.

摘要: 高级持续性威胁（APT）是技术精湛的对手发起的长期、秘密入侵，这些入侵会危及高价值系统以窃取数据或扰乱运营。从大量、异类的日志中重建完整的攻击链对于有效的攻击调查至关重要，但现有方法存在平台通用性较差、对不断发展的策略的概括性有限以及无法生成可供分析师使用的报告的问题。大型语言模型（LLM）提供强大的语义理解和总结能力，但在该领域，它们很难捕捉对准确重建至关重要的长期、跨日志依赖关系。   为了解决这些问题，我们提出了一个LLM授权的攻击调查框架，该框架增强了动态自适应的杀戮链对齐威胁知识库。我们将与攻击相关的行为组织到富含语义注释的阶段感知知识单元中，使LLM能够迭代地检索相关情报、执行因果推理并逐步扩展调查上下文。该过程重建多阶段攻击场景并生成连贯、人类可读的调查报告。评估了跨Windows和Linux的单主机和多主机环境的15种攻击场景（超过430万个日志事件，7.2 GB数据），该系统的平均真阳性率（TPR）为97.1%，平均假阳性率（FPR）为0.2%，显着优于SOTA方法ATLAS，平均TPR为79.2%，平均FPR为29.1%。



## **10. Geometric origin of adversarial vulnerability in deep learning**

深度学习中对抗脆弱性的几何起源 cs.LG

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01235v1) [paper-pdf](http://arxiv.org/pdf/2509.01235v1)

**Authors**: Yixiong Ren, Wenkang Du, Jianhui Zhou, Haiping Huang

**Abstract**: How to balance training accuracy and adversarial robustness has become a challenge since the birth of deep learning. Here, we introduce a geometry-aware deep learning framework that leverages layer-wise local training to sculpt the internal representations of deep neural networks. This framework promotes intra-class compactness and inter-class separation in feature space, leading to manifold smoothness and adversarial robustness against white or black box attacks. The performance can be explained by an energy model with Hebbian coupling between elements of the hidden representation. Our results thus shed light on the physics of learning in the direction of alignment between biological and artificial intelligence systems. Using the current framework, the deep network can assimilate new information into existing knowledge structures while reducing representation interference.

摘要: 自深度学习诞生以来，如何平衡训练准确性和对抗鲁棒性就成为一个挑战。在这里，我们引入了一个几何感知的深度学习框架，该框架利用分层本地训练来塑造深度神经网络的内部表示。该框架促进了特征空间中的类内紧凑性和类间分离，从而实现了对白盒或黑盒攻击的多维平滑性和对抗鲁棒性。性能可以通过隐藏表示元素之间具有赫布式耦合的能量模型来解释。因此，我们的结果揭示了生物和人工智能系统之间协调方向的学习物理学。使用当前框架，深度网络可以将新信息吸收到现有知识结构中，同时减少表示干扰。



## **11. Crosstalk Attacks and Defence in a Shared Quantum Computing Environment**

共享量子计算环境中的串话攻击和防御 quant-ph

13 pages, 7 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2402.02753v2) [paper-pdf](http://arxiv.org/pdf/2402.02753v2)

**Authors**: Benjamin Harper, Behnam Tonekaboni, Bahar Goldozian, Martin Sevior, Muhammad Usman

**Abstract**: Quantum computing has the potential to provide solutions to problems that are intractable on classical computers, but the accuracy of the current generation of quantum computers suffer from the impact of noise or errors such as leakage, crosstalk, dephasing, and amplitude damping among others. As the access to quantum computers is almost exclusively in a shared environment through cloud-based services, it is possible that an adversary can exploit crosstalk noise to disrupt quantum computations on nearby qubits, even carefully designing quantum circuits to purposely lead to wrong answers. In this paper, we analyze the extent and characteristics of crosstalk noise through tomography conducted on IBM Quantum computers, leading to an enhanced crosstalk simulation model. Our results indicate that crosstalk noise is a significant source of errors on IBM quantum hardware, making crosstalk based attack a viable threat to quantum computing in a shared environment. Based on our crosstalk simulator benchmarked against IBM hardware, we assess the impact of crosstalk attacks and develop strategies for mitigating crosstalk effects. Through a systematic set of simulations, we assess the effectiveness of three crosstalk attack mitigation strategies, namely circuit separation, qubit allocation optimization via reinforcement learning, and the use of spectator qubits, and show that they all overcome crosstalk attacks with varying degrees of success and help to secure quantum computing in a shared platform.

摘要: 量子计算有潜力为经典计算机上棘手的问题提供解决方案，但当前一代量子计算机的准确性受到泄漏、串话、去相和幅度衰减等噪音或误差的影响。由于量子计算机的访问几乎完全是在共享环境中通过基于云的服务进行的，因此对手可能会利用串话噪音来破坏附近量子位上的量子计算，甚至精心设计量子电路来故意导致错误的答案。在本文中，我们通过在IBM Quantum计算机上进行的断层扫描分析了串话噪音的程度和特征，从而建立了增强的串话模拟模型。我们的结果表明，串话噪音是IBM量子硬件上错误的重要来源，使得基于串话的攻击成为共享环境中量子计算的可行威胁。基于我们以IBM硬件为基准的串话模拟器，我们评估串话攻击的影响并开发减轻串话影响的策略。通过一组系统的模拟，我们评估了三种串话攻击缓解策略的有效性，即电路分离、通过强化学习进行量子位分配优化以及使用旁观量子位，并表明它们都以不同程度的成功克服了串话攻击，并有助于在共享平台中保护量子计算。



## **12. Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation**

克隆无法窃取的内容：通过Logit泄漏和蒸馏进行黑匣子LLM复制 cs.CR

8 pages. Accepted for publication in the proceedings of 7th IEEE  International Conference on Trust, Privacy and Security in Intelligent  Systems, and Applications (IEEE TPS 2025)

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00973v1) [paper-pdf](http://arxiv.org/pdf/2509.00973v1)

**Authors**: Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments.

摘要: 大型语言模型（LLM）越来越多地部署在关键任务系统中，促进卫星操作、指挥与控制、军事决策支持和网络防御等任务。其中许多系统都是通过应用程序编程接口（API）访问的。当此类API缺乏强大的访问控制时，它们可能会暴露完整或顶级k日志，从而创建一个重要且经常被忽视的攻击面。现有技术主要集中在重建输出投影层或提取表面级行为。然而，在严格的查询约束下重新生成黑匣子模型仍然没有得到充分的探索。我们通过引入一个受约束的复制管道来解决这一差距，该管道将部分logit泄漏转换为功能性可部署的替代模型克隆。我们的两阶段方法（i）通过对logit进行奇异值分解（DID）从10 k以下的黑匣子查询中收集前k个logit来重建输出投影矩阵，然后（ii）将剩余的架构提炼成具有不同Transformer深度的紧凑学生模型，在开源数据集上训练。一个6层的学生可以重建97.6%的6层教师模型的隐藏状态几何，而困惑度只增加了7.31%，负对数似然（NLL）为7.58。4层变体在相当的性能下实现了17.1%的推理速度和18.1%的参数减少。整个攻击只需不到24个图形处理单元（图形处理单元）小时即可完成，并避免触发API速率限制防御。这些结果展示了成本有限的对手可以多快地克隆LLM，凸显了对强化推理API和安全本地防御部署的迫切需求。



## **13. FusionCounting: Robust visible-infrared image fusion guided by crowd counting via multi-task learning**

FusionCounting：通过多任务学习由人群计数引导的鲁棒可见光-红外图像融合 cs.CV

11 pages, 9 figures

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2508.20817v2) [paper-pdf](http://arxiv.org/pdf/2508.20817v2)

**Authors**: He Li, Xinyu Liu, Weihang Kong, Xingchen Zhang

**Abstract**: Visible and infrared image fusion (VIF) is an important multimedia task in computer vision. Most VIF methods focus primarily on optimizing fused image quality. Recent studies have begun incorporating downstream tasks, such as semantic segmentation and object detection, to provide semantic guidance for VIF. However, semantic segmentation requires extensive annotations, while object detection, despite reducing annotation efforts compared with segmentation, faces challenges in highly crowded scenes due to overlapping bounding boxes and occlusion. Moreover, although RGB-T crowd counting has gained increasing attention in recent years, no studies have integrated VIF and crowd counting into a unified framework. To address these challenges, we propose FusionCounting, a novel multi-task learning framework that integrates crowd counting into the VIF process. Crowd counting provides a direct quantitative measure of population density with minimal annotation, making it particularly suitable for dense scenes. Our framework leverages both input images and population density information in a mutually beneficial multi-task design. To accelerate convergence and balance tasks contributions, we introduce a dynamic loss function weighting strategy. Furthermore, we incorporate adversarial training to enhance the robustness of both VIF and crowd counting, improving the model's stability and resilience to adversarial attacks. Experimental results on public datasets demonstrate that FusionCounting not only enhances image fusion quality but also achieves superior crowd counting performance.

摘要: 可见光和红外图像融合（VIF）是计算机视觉中一项重要的多媒体任务。大多数VIF方法主要关注优化融合图像质量。最近的研究已经开始整合下游任务，例如语义分割和对象检测，为VIF提供语义指导。然而，语义分割需要大量的注释，而对象检测尽管与分割相比减少了注释工作，但由于重叠的边界框和遮挡，在高度拥挤的场景中面临挑战。此外，尽管近年来RGB-T人群计数越来越受到关注，但还没有研究将VIF和人群计数整合到统一的框架中。为了应对这些挑战，我们提出FusionCounting，这是一种新型的多任务学习框架，可将人群计数集成到VIF流程中。人群计数提供了人口密度的直接定量测量，只需最少的注释，使其特别适合密集场景。我们的框架在互利的多任务设计中利用输入图像和人口密度信息。为了加速收敛和平衡任务贡献，我们引入了动态损失函数加权策略。此外，我们结合了对抗性训练来增强VIF和人群计数的鲁棒性，提高了模型的稳定性和对抗性攻击的弹性。在公开数据集上的实验结果表明，FusionCounting不仅提高了图像融合质量，而且实现了优越的人群计数性能。



## **14. Redesigning Traffic Signs to Mitigate Machine-Learning Patch Attacks**

重新设计交通标志以缓解机器学习补丁攻击 cs.CR

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2402.04660v3) [paper-pdf](http://arxiv.org/pdf/2402.04660v3)

**Authors**: Tsufit Shua, Liron David, Mahmood Sharif

**Abstract**: Traffic-Sign Recognition (TSR) is a critical safety component for autonomous driving. Unfortunately, however, past work has highlighted the vulnerability of TSR models to physical-world attacks, through low-cost, easily deployable adversarial patches leading to misclassification. To mitigate these threats, most defenses focus on altering the training process or modifying the inference procedure. Still, while these approaches improve adversarial robustness, TSR remains susceptible to attacks attaining substantial success rates.   To further the adversarial robustness of TSR, this work offers a novel approach that redefines traffic-sign designs to create signs that promote robustness while remaining interpretable to humans. Our framework takes three inputs: (1) A traffic-sign standard along with modifiable features and associated constraints; (2) A state-of-the-art adversarial training method; and (3) A function for efficiently synthesizing realistic traffic-sign images. Using these user-defined inputs, the framework emits an optimized traffic-sign standard such that traffic signs generated per this standard enable training TSR models with increased adversarial robustness.   We evaluate the effectiveness of our framework via a concrete implementation, where we allow modifying the pictograms (i.e., symbols) and colors of traffic signs. The results show substantial improvements in robustness -- with gains of up to 16.33%--24.58% in robust accuracy over state-of-the-art methods -- while benign accuracy is even improved. Importantly, a user study also confirms that the redesigned traffic signs remain easily recognizable and to human observers. Overall, the results highlight that carefully redesigning traffic signs can significantly enhance TSR system robustness without compromising human interpretability.

摘要: 手势识别（TSB）是自动驾驶的关键安全组件。然而不幸的是，过去的工作强调了TSB模型对物理世界攻击的脆弱性，因为低成本、易于部署的对抗补丁导致了错误分类。为了减轻这些威胁，大多数防御措施都集中在改变训练过程或修改推理程序上。尽管如此，虽然这些方法提高了对抗鲁棒性，但TSB仍然容易受到获得相当大成功率的攻击。   为了进一步增强TSB的对抗鲁棒性，这项工作提供了一种新的方法，重新定义交通标志设计，以创建既能提高鲁棒性又能为人类解释的标志。我们的框架需要三个输入：（1）交通标志标准以及可修改的特征和相关的约束;（2）最先进的对抗训练方法;（3）用于有效合成真实交通标志图像的功能。使用这些用户定义的输入，该框架发布优化的交通标志标准，以便按照该标准生成的交通标志能够训练具有更高的对抗鲁棒性的TSB模型。   我们通过具体实现来评估框架的有效性，其中我们允许修改象形图（即，符号）和交通标志的颜色。结果显示鲁棒性有了大幅提高--鲁棒准确性比最先进的方法提高了16.33%--24.58%--而良性准确性甚至得到了提高。重要的是，一项用户研究还证实，重新设计的交通标志仍然容易被人类观察者识别。总体而言，结果强调，仔细重新设计交通标志可以显着增强TSB系统的稳健性，而不会损害人类的解释性。



## **15. Sequential Difference Maximization: Generating Adversarial Examples via Multi-Stage Optimization**

序列差异最大化：通过多阶段优化生成对抗性示例 cs.CV

5 pages, 2 figures, 5 tables, CIKM 2025

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00826v1) [paper-pdf](http://arxiv.org/pdf/2509.00826v1)

**Authors**: Xinlei Liu, Tao Hu, Peng Yi, Weitao Han, Jichao Xie, Baolin Li

**Abstract**: Efficient adversarial attack methods are critical for assessing the robustness of computer vision models. In this paper, we reconstruct the optimization objective for generating adversarial examples as "maximizing the difference between the non-true labels' probability upper bound and the true label's probability," and propose a gradient-based attack method termed Sequential Difference Maximization (SDM). SDM establishes a three-layer optimization framework of "cycle-stage-step." The processes between cycles and between iterative steps are respectively identical, while optimization stages differ in terms of loss functions: in the initial stage, the negative probability of the true label is used as the loss function to compress the solution space; in subsequent stages, we introduce the Directional Probability Difference Ratio (DPDR) loss function to gradually increase the non-true labels' probability upper bound by compressing the irrelevant labels' probabilities. Experiments demonstrate that compared with previous SOTA methods, SDM not only exhibits stronger attack performance but also achieves higher attack cost-effectiveness. Additionally, SDM can be combined with adversarial training methods to enhance their defensive effects. The code is available at https://github.com/X-L-Liu/SDM.

摘要: 有效的对抗攻击方法对于评估计算机视觉模型的稳健性至关重要。本文将生成对抗性示例的优化目标重建为“最大化非真标签概率上限和真标签概率之间的差异”，并提出了一种基于梯度的攻击方法，称为序列差异最大化（Sendential Difference Maximation）。SDP建立了“周期-阶段-步骤”的三层优化框架。“循环之间和迭代步骤之间的过程分别相同，而优化阶段在损失函数方面有所不同：在初始阶段，使用真实标签的负概率作为损失函数来压缩解空间;在随后的阶段，我们引入方向概率差比（DPDR）损失函数来逐步增加非通过压缩不相关标签的概率来确定真实标签的概率上限。实验表明，与之前的SOTA方法相比，Sends不仅具有更强的攻击性能，而且具有更高的攻击性价比。此外，SDM可以与对抗性训练方法相结合，以增强其防御效果。该代码可在https://github.com/X-L-Liu/SDM上获取。



## **16. Membership Inference Attacks on Large-Scale Models: A Survey**

对大规模模型的成员推断攻击：一项调查 cs.LG

Preprint. Submitted for peer review. The final version may differ

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2503.19338v3) [paper-pdf](http://arxiv.org/pdf/2503.19338v3)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: As large-scale models such as Large Language Models (LLMs) and Large Multimodal Models (LMMs) see increasing deployment, their privacy risks remain underexplored. Membership Inference Attacks (MIAs), which reveal whether a data point was used in training the target model, are an important technique for exposing or assessing privacy risks and have been shown to be effective across diverse machine learning algorithms. However, despite extensive studies on MIAs in classic models, there remains a lack of systematic surveys addressing their effectiveness and limitations in large-scale models. To address this gap, we provide the first comprehensive review of MIAs targeting LLMs and LMMs, analyzing attacks by model type, adversarial knowledge, and strategy. Unlike prior surveys, we further examine MIAs across multiple stages of the model pipeline, including pre-training, fine-tuning, alignment, and Retrieval-Augmented Generation (RAG). Finally, we identify open challenges and propose future research directions for strengthening privacy resilience in large-scale models.

摘要: 随着大型语言模型（LLM）和大型多模式模型（LSYS）等大规模模型的部署不断增加，但其隐私风险仍然未得到充分研究。成员推断攻击（MIA）揭示了数据点是否用于训练目标模型，是暴露或评估隐私风险的重要技术，并已被证明在各种机器学习算法中有效。然而，尽管对经典模型中的MIA进行了广泛的研究，但仍然缺乏系统性的调查来解决其在大规模模型中的有效性和局限性。为了弥补这一差距，我们首次对针对LLM和LSYS的MIA进行了全面审查，按模型类型、对抗性知识和策略分析攻击。与之前的调查不同，我们进一步检查了模型管道的多个阶段的MIA，包括预训练、微调、对齐和检索增强生成（RAG）。最后，我们确定了开放的挑战，并提出了未来的研究方向，以加强大规模模型中的隐私弹性。



## **17. Bayesian and Multi-Objective Decision Support for Real-Time Cyber-Physical Incident Mitigation**

实时网络物理事件缓解的Bayesian和多目标决策支持 cs.CR

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00770v1) [paper-pdf](http://arxiv.org/pdf/2509.00770v1)

**Authors**: Shaofei Huang, Christopher M. Poskitt, Lwin Khin Shar

**Abstract**: This research proposes a real-time, adaptive decision-support framework for mitigating cyber incidents in cyber-physical systems, developed in response to an increasing reliance on these systems within critical infrastructure and evolving adversarial tactics. Existing decision-support systems often fall short in accounting for multi-agent, multi-path attacks and trade-offs between safety and operational continuity. To address this, our framework integrates hierarchical system modelling with Bayesian probabilistic reasoning, constructing Bayesian Network Graphs from system architecture and vulnerability data. Models are encoded using a Domain Specific Language to enhance computational efficiency and support dynamic updates. In our approach, we use a hybrid exposure probability estimation framework, which combines Exploit Prediction Scoring System and Common Vulnerability Scoring System scores via Bayesian confidence calibration to handle epistemic uncertainty caused by incomplete or heterogeneous vulnerability metadata. Mitigation recommendations are generated as countermeasure portfolios, refined using multi-objective optimisation to identify Pareto-optimal strategies balancing attack likelihood, impact severity, and system availability. To accommodate time- and resource-constrained incident response, frequency-based heuristics are applied to prioritise countermeasures across the optimised portfolios. The framework was evaluated through three representative cyber-physical attack scenarios, demonstrating its versatility in handling complex adversarial behaviours under real-time response constraints. The results affirm its utility in operational contexts and highlight the robustness of our proposed approach across diverse threat environments.

摘要: 这项研究提出了一种实时、自适应的决策支持框架，用于缓解网络物理系统中的网络事件，该框架是为了应对关键基础设施内对这些系统日益依赖和不断发展的对抗策略而开发的。现有的决策支持系统往往无法应对多代理、多路径攻击以及安全性和运营连续性之间的权衡。为了解决这个问题，我们的框架将分层系统建模与Bayesian概率推理集成，从系统架构和漏洞数据构建Bayesian网络图。模型使用领域特定语言进行编码，以提高计算效率并支持动态更新。在我们的方法中，我们使用混合暴露概率估计框架，该框架通过Bayesian置信度校准将利用预测评分系统和通用脆弱性评分系统评分结合起来，以处理不完整或异类脆弱性元数据造成的认识不确定性。缓解建议作为对策组合生成，并使用多目标优化进行完善，以识别平衡攻击可能性、影响严重性和系统可用性的帕累托最优策略。为了适应时间和资源受限的事件响应，应用基于频率的试探法来在优化的投资组合中确定对策的优先顺序。该框架通过三种代表性的网络物理攻击场景进行了评估，展示了其在实时响应约束下处理复杂对抗行为的多功能性。结果证实了它在操作环境中的实用性，并强调了我们提出的方法在不同威胁环境中的稳健性。



## **18. Enabling Trustworthy Federated Learning via Remote Attestation for Mitigating Byzantine Threats**

通过远程认证启用值得信赖的联邦学习以缓解拜占庭威胁 cs.CR

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00634v1) [paper-pdf](http://arxiv.org/pdf/2509.00634v1)

**Authors**: Chaoyu Zhang, Heng Jin, Shanghao Shi, Hexuan Yu, Sydney Johns, Y. Thomas Hou, Wenjing Lou

**Abstract**: Federated Learning (FL) has gained significant attention for its privacy-preserving capabilities, enabling distributed devices to collaboratively train a global model without sharing raw data. However, its distributed nature forces the central server to blindly trust the local training process and aggregate uncertain model updates, making it susceptible to Byzantine attacks from malicious participants, especially in mission-critical scenarios. Detecting such attacks is challenging due to the diverse knowledge across clients, where variations in model updates may stem from benign factors, such as non-IID data, rather than adversarial behavior. Existing data-driven defenses struggle to distinguish malicious updates from natural variations, leading to high false positive rates and poor filtering performance.   To address this challenge, we propose Sentinel, a remote attestation (RA)-based scheme for FL systems that regains client-side transparency and mitigates Byzantine attacks from a system security perspective. Our system employs code instrumentation to track control-flow and monitor critical variables in the local training process. Additionally, we utilize a trusted training recorder within a Trusted Execution Environment (TEE) to generate an attestation report, which is cryptographically signed and securely transmitted to the server. Upon verification, the server ensures that legitimate client training processes remain free from program behavior violation or data manipulation, allowing only trusted model updates to be aggregated into the global model. Experimental results on IoT devices demonstrate that Sentinel ensures the trustworthiness of the local training integrity with low runtime and memory overhead.

摘要: 联合学习（FL）因其隐私保护功能而受到广泛关注，使分布式设备能够在无需共享原始数据的情况下协作训练全球模型。然而，其分布式性质迫使中央服务器盲目信任本地训练过程并聚合不确定的模型更新，使其容易受到恶意参与者的拜占庭攻击，尤其是在关键任务场景中。由于客户之间的知识多样化，检测此类攻击具有挑战性，其中模型更新的差异可能源于良性因素（例如非IID数据），而不是对抗行为。现有的数据驱动防御很难区分恶意更新与自然变化，导致误报率高和过滤性能差。   为了应对这一挑战，我们提出了Sentinel，这是一种针对FL系统的基于远程认证（RA）的方案，可以恢复客户端透明度并从系统安全角度减轻拜占庭攻击。我们的系统采用代码工具来跟踪控制流并监控本地培训过程中的关键变量。此外，我们利用可信执行环境（TEK）中的可信训练记录器来生成证明报告，该报告经过加密签名并安全传输到服务器。验证后，服务器确保合法的客户端训练流程不受程序行为违规或数据操纵的影响，仅允许将受信任的模型更新聚合到全局模型中。物联网设备上的实验结果表明，Sentinel以较低的运行时间和内存负担确保了本地训练完整性的可信度。



## **19. On the Thermal Vulnerability of 3D-Stacked High-Bandwidth Memory Architectures**

3D堆叠高带宽存储器架构的热脆弱性 cs.AR

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00633v1) [paper-pdf](http://arxiv.org/pdf/2509.00633v1)

**Authors**: Mehdi Elahi, Mohamed R. Elshamy, Abdel-Hameed A. Badawy, Ahmad Patooghy

**Abstract**: 3D-stacked High Bandwidth Memory (HBM) architectures provide high-performance memory interactions to address the well-known performance challenge, namely the memory wall. However, these architectures are susceptible to thermal vulnerabilities due to the inherent vertical adjacency that occurs during the manufacturing process of HBM architectures. We anticipate that adversaries may exploit the intense vertical and lateral adjacency to design and develop thermal performance degradation attacks on the memory banks that host data/instructions from victim applications. In such attacks, the adversary manages to inject short and intense heat pulses from vertically and/or laterally adjacent memory banks, creating a convergent thermal wave that maximizes impact and delays the victim application from accessing its data/instructions. As the attacking application does not access any out-of-range memory locations, it can bypass both design-time security tests and the operating system's memory management policies. In other words, since the attack mimics legitimate workloads, it will be challenging to detect.

摘要: 3D堆叠高带宽内存（HBM）架构提供高性能内存交互，以解决众所周知的性能挑战，即内存墙。然而，由于HBM架构制造过程中出现的固有垂直邻近，这些架构容易受到热漏洞的影响。我们预计对手可能会利用强烈的垂直和横向邻近性来设计和开发对托管来自受害应用程序的数据/指令的存储库的热性能降级攻击。在此类攻击中，对手设法从垂直和/或横向相邻的存储体注入短而强的热脉冲，从而创建收敛热波，最大限度地发挥影响并延迟受害应用程序访问其数据/指令。由于攻击应用程序不会访问任何超出范围的内存位置，因此它可以绕过设计时安全测试和操作系统的内存管理策略。换句话说，由于该攻击模仿了合法的工作负载，因此检测起来很困难。



## **20. FedThief: Harming Others to Benefit Oneself in Self-Centered Federated Learning**

FedThief：在以自我为中心的联邦学习中伤害他人以造福自己 cs.LG

12 pages, 5 figures

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00540v1) [paper-pdf](http://arxiv.org/pdf/2509.00540v1)

**Authors**: Xiangyu Zhang, Mang Ye

**Abstract**: In federated learning, participants' uploaded model updates cannot be directly verified, leaving the system vulnerable to malicious attacks. Existing attack strategies have adversaries upload tampered model updates to degrade the global model's performance. However, attackers also degrade their own private models, gaining no advantage. In real-world scenarios, attackers are driven by self-centered motives: their goal is to gain a competitive advantage by developing a model that outperforms those of other participants, not merely to cause disruption. In this paper, we study a novel Self-Centered Federated Learning (SCFL) attack paradigm, in which attackers not only degrade the performance of the global model through attacks but also enhance their own models within the federated learning process. We propose a framework named FedThief, which degrades the performance of the global model by uploading modified content during the upload stage. At the same time, it enhances the private model's performance through divergence-aware ensemble techniques, where "divergence" quantifies the deviation between private and global models, that integrate global updates and local knowledge. Extensive experiments show that our method effectively degrades the global model performance while allowing the attacker to obtain an ensemble model that significantly outperforms the global model.

摘要: 在联邦学习中，参与者上传的模型更新无法直接验证，导致系统容易受到恶意攻击。现有的攻击策略让对手上传篡改的模型更新以降低全局模型的性能。然而，攻击者也会降级他们自己的私有模型，没有获得任何优势。在现实世界的场景中，攻击者受到以自我为中心的动机的驱动：他们的目标是通过开发一种优于其他参与者的模型来获得竞争优势，而不仅仅是为了造成破坏。本文研究了一种新型的自中心联邦学习（SCFL）攻击范式，其中攻击者不仅通过攻击降低全局模型的性能，而且还在联邦学习过程中增强自己的模型。我们提出了一个名为FedThief的框架，该框架通过在上传阶段上传修改后的内容来降低全局模型的性能。与此同时，它通过分歧感知集成技术增强了私有模型的性能，其中“分歧”量化了私有模型和全球模型之间的偏差，集成了全球更新和本地知识。大量实验表明，我们的方法有效地降低了全局模型的性能，同时允许攻击者获得显着优于全局模型的集成模型。



## **21. Similarity between Units of Natural Language: The Transition from Coarse to Fine Estimation**

自然语言单位之间的相似性：从粗略估计到精细估计的过渡 cs.CL

PhD thesis

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2210.14275v2) [paper-pdf](http://arxiv.org/pdf/2210.14275v2)

**Authors**: Wenchuan Mu

**Abstract**: Capturing the similarities between human language units is crucial for explaining how humans associate different objects, and therefore its computation has received extensive attention, research, and applications. With the ever-increasing amount of information around us, calculating similarity becomes increasingly complex, especially in many cases, such as legal or medical affairs, measuring similarity requires extra care and precision, as small acts within a language unit can have significant real-world effects. My research goal in this thesis is to develop regression models that account for similarities between language units in a more refined way.   Computation of similarity has come a long way, but approaches to debugging the measures are often based on continually fitting human judgment values. To this end, my goal is to develop an algorithm that precisely catches loopholes in a similarity calculation. Furthermore, most methods have vague definitions of the similarities they compute and are often difficult to interpret. The proposed framework addresses both shortcomings. It constantly improves the model through catching different loopholes. In addition, every refinement of the model provides a reasonable explanation. The regression model introduced in this thesis is called progressively refined similarity computation, which combines attack testing with adversarial training. The similarity regression model of this thesis achieves state-of-the-art performance in handling edge cases.

摘要: 捕捉人类语言单位之间的相似性对于解释人类如何将不同对象关联起来至关重要，因此其计算受到了广泛的关注、研究和应用。随着我们周围的信息量不断增加，计算相似性变得越来越复杂，特别是在许多情况下，例如法律或医疗事务，测量相似性需要格外小心和精确，因为语言单位内的小行为可能会产生显着的现实世界影响。我在这篇论文中的研究目标是开发回归模型，以更精确的方式考虑语言单位之间的相似性。   相似性的计算已经取得了很大的进步，但调试测量的方法通常基于不断匹配的人类判断值。为此，我的目标是开发一种精确捕捉相似度计算中漏洞的算法。此外，大多数方法对其计算的相似性的定义模糊，并且通常难以解释。拟议的框架解决了这两个缺点。它通过捕捉不同的漏洞来不断改进模型。此外，模型的每一次细化都提供了合理的解释。本文引入的回归模型称为渐进式细化相似度计算，它将攻击测试与对抗训练相结合。本文的相似度回归模型在处理边缘案例方面达到了最先进的性能。



## **22. The Resurgence of GCG Adversarial Attacks on Large Language Models**

GCG对大型语言模型的对抗性攻击的卷土重来 cs.CL

12 pages, 5 figures

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00391v1) [paper-pdf](http://arxiv.org/pdf/2509.00391v1)

**Authors**: Yuting Tan, Xuying Li, Zhuo Li, Huizhen Shu, Peikang Hu

**Abstract**: Gradient-based adversarial prompting, such as the Greedy Coordinate Gradient (GCG) algorithm, has emerged as a powerful method for jailbreaking large language models (LLMs). In this paper, we present a systematic appraisal of GCG and its annealing-augmented variant, T-GCG, across open-source LLMs of varying scales. Using Qwen2.5-0.5B, LLaMA-3.2-1B, and GPT-OSS-20B, we evaluate attack effectiveness on both safety-oriented prompts (AdvBench) and reasoning-intensive coding prompts. Our study reveals three key findings: (1) attack success rates (ASR) decrease with model size, reflecting the increasing complexity and non-convexity of larger models' loss landscapes; (2) prefix-based heuristics substantially overestimate attack effectiveness compared to GPT-4o semantic judgments, which provide a stricter and more realistic evaluation; and (3) coding-related prompts are significantly more vulnerable than adversarial safety prompts, suggesting that reasoning itself can be exploited as an attack vector. In addition, preliminary results with T-GCG show that simulated annealing can diversify adversarial search and achieve competitive ASR under prefix evaluation, though its benefits under semantic judgment remain limited. Together, these findings highlight the scalability limits of GCG, expose overlooked vulnerabilities in reasoning tasks, and motivate further development of annealing-inspired strategies for more robust adversarial evaluation.

摘要: 基于对象的对抗提示，例如贪婪坐标梯度（GCG）算法，已成为越狱大型语言模型（LLM）的强大方法。在本文中，我们在不同规模的开源LLM中对GCG及其软化增强变体T-GCG进行了系统评估。使用Qwen 2.5 -0.5B、LLaMA-3.2-1B和GPT-OSS-20 B，我们评估了面向安全的提示（AdvBench）和推理密集型编码提示的攻击有效性。我们的研究揭示了三个关键发现：（1）攻击成功率（ASB）随着模型大小的增加而降低，反映了较大模型损失景观的复杂性和非凸性的增加;（2）与GPT-4 o语义判断相比，基于后缀的启发式算法大大高估了攻击有效性，这提供了更严格、更现实的评估;（3）与编码相关的提示比对抗性安全提示明显更容易受到攻击，这表明推理本身可以被用作攻击载体。此外，T-GCG的初步结果表明，模拟排序可以使对抗性搜索多样化，并在前置评估下实现有竞争力的ASB，尽管其在语义判断下的好处仍然有限。总而言之，这些发现凸显了GCG的可扩展性限制，暴露了推理任务中被忽视的漏洞，并激励进一步开发受退变启发的策略，以实现更稳健的对抗性评估。



## **23. Unifying Adversarial Perturbation for Graph Neural Networks**

图神经网络的统一对抗扰动 cs.LG

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00387v1) [paper-pdf](http://arxiv.org/pdf/2509.00387v1)

**Authors**: Jinluan Yang, Ruihao Zhang, Zhengyu Chen, Fei Wu, Kun Kuang

**Abstract**: This paper studies the vulnerability of Graph Neural Networks (GNNs) to adversarial attacks on node features and graph structure. Various methods have implemented adversarial training to augment graph data, aiming to bolster the robustness and generalization of GNNs. These methods typically involve applying perturbations to the node feature, weights, or graph structure and subsequently minimizing the loss by learning more robust graph model parameters under the adversarial perturbations. Despite the effectiveness of adversarial training in enhancing GNNs' robustness and generalization abilities, its application has been largely confined to specific datasets and GNN types. In this paper, we propose a novel method, PerturbEmbedding, that integrates adversarial perturbation and training, enhancing GNNs' resilience to such attacks and improving their generalization ability. PerturbEmbedding performs perturbation operations directly on every hidden embedding of GNNs and provides a unified framework for most existing perturbation strategies/methods. We also offer a unified perspective on the forms of perturbations, namely random and adversarial perturbations. Through experiments on various datasets using different backbone models, we demonstrate that PerturbEmbedding significantly improves both the robustness and generalization abilities of GNNs, outperforming existing methods. The rejection of both random (non-targeted) and adversarial (targeted) perturbations further enhances the backbone model's performance.

摘要: 本文研究了图神经网络（GNN）对节点特征和图结构的对抗攻击的脆弱性。各种方法都实施了对抗训练来增强图形数据，旨在增强GNN的鲁棒性和通用性。这些方法通常涉及对节点特征、权重或图结构应用扰动，然后通过在对抗性扰动下学习更稳健的图模型参数来最大限度地减少损失。尽管对抗训练在增强GNN的稳健性和概括能力方面有效，但其应用在很大程度上仅限于特定的数据集和GNN类型。本文中，我们提出了一种新型方法PerturbEmbedding，该方法集成了对抗性扰动和训练，增强GNN对此类攻击的弹性并提高其概括能力。PerturbEmbedding直接对GNN的每个隐藏嵌入执行扰动操作，并为大多数现有的扰动策略/方法提供统一的框架。我们还提供了关于扰动形式的统一视角，即随机扰动和对抗扰动。通过使用不同主干模型对各种数据集进行实验，我们证明PerturbEmbedding显着提高了GNN的鲁棒性和概括能力，优于现有方法。拒绝随机（非目标）和对抗（目标）扰动进一步增强了主干模型的性能。



## **24. Activation Steering Meets Preference Optimization: Defense Against Jailbreaks in Vision Language Models**

激活引导满足偏好优化：视觉语言模型中的越狱防御 cs.CV

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00373v1) [paper-pdf](http://arxiv.org/pdf/2509.00373v1)

**Authors**: Sihao Wu, Gaojie Jin, Wei Huang, Jianhong Wang, Xiaowei Huang

**Abstract**: Vision Language Models (VLMs) have demonstrated impressive capabilities in integrating visual and textual information for understanding and reasoning, but remain highly vulnerable to adversarial attacks. While activation steering has emerged as a promising defence, existing approaches often rely on task-specific contrastive prompts to extract harmful directions, which exhibit suboptimal performance and can degrade visual grounding performance. To address these limitations, we propose \textit{Sequence-Level Preference Optimization} for VLM (\textit{SPO-VLM}), a novel two-stage defense framework that combines activation-level intervention with policy-level optimization to enhance model robustness. In \textit{Stage I}, we compute adaptive layer-specific steering vectors from diverse data sources, enabling generalized suppression of harmful behaviors during inference. In \textit{Stage II}, we refine these steering vectors through a sequence-level preference optimization process. This stage integrates automated toxicity assessment, as well as visual-consistency rewards based on caption-image alignment, to achieve safe and semantically grounded text generation. The two-stage structure of SPO-VLM balances efficiency and effectiveness by combining a lightweight mitigation foundation in Stage I with deeper policy refinement in Stage II. Extensive experiments shown SPO-VLM enhances safety against attacks via activation steering and preference optimization, while maintaining strong performance on benign tasks without compromising visual understanding capabilities. We will release our code, model weights, and evaluation toolkit to support reproducibility and future research. \textcolor{red}{Warning: This paper may contain examples of offensive or harmful text and images.}

摘要: 视觉语言模型（VLM）在集成视觉和文本信息以进行理解和推理方面表现出了令人印象深刻的能力，但仍然极易受到对抗性攻击。虽然激活引导已成为一种有希望的防御手段，但现有的方法通常依赖于特定于任务的对比提示来提取有害的方向，这些方向表现出次优的性能，并且可能会降低视觉基础性能。为了解决这些限制，我们提出了VLM的\textit{Sequence-Level PreferenceOptimizer}（\textit{SPO-VLM}），这是一种新型的两阶段防御框架，将激活级别干预与策略级别优化相结合，以增强模型稳健性。在\texttit {Stage I}中，我们从不同的数据源计算自适应的特定于层的引导载体，从而能够在推理期间对有害行为进行广义抑制。在\texttit {Stage II}中，我们通过序列级偏好优化过程来细化这些引导载体。该阶段集成了自动毒性评估以及基于字幕图像对齐的视觉一致性奖励，以实现安全且基于语义的文本生成。SPO-VLM的两阶段结构通过将第一阶段的轻量级缓解基础与第二阶段的更深入的政策细化相结合，平衡了效率和有效性。大量实验表明，SPO-VLM通过激活引导和偏好优化增强了针对攻击的安全性，同时在良性任务上保持强劲的性能，而不会损害视觉理解能力。我们将发布我们的代码、模型权重和评估工具包，以支持可重复性和未来的研究。\textColor{red}{警告：本文可能包含冒犯性或有害文本和图像的示例。}



## **25. MorphGen: Morphology-Guided Representation Learning for Robust Single-Domain Generalization in Histopathological Cancer Classification**

MorphGen：形态学引导的表示学习，用于组织病理学癌症分类中的鲁棒单域概括 cs.CV

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00311v1) [paper-pdf](http://arxiv.org/pdf/2509.00311v1)

**Authors**: Hikmat Khan, Syed Farhan Alam Zaidi, Pir Masoom Shah, Kiruthika Balakrishnan, Rabia Khan, Muhammad Waqas, Jia Wu

**Abstract**: Domain generalization in computational histopathology is hindered by heterogeneity in whole slide images (WSIs), caused by variations in tissue preparation, staining, and imaging conditions across institutions. Unlike machine learning systems, pathologists rely on domain-invariant morphological cues such as nuclear atypia (enlargement, irregular contours, hyperchromasia, chromatin texture, spatial disorganization), structural atypia (abnormal architecture and gland formation), and overall morphological atypia that remain diagnostic across diverse settings. Motivated by this, we hypothesize that explicitly modeling biologically robust nuclear morphology and spatial organization will enable the learning of cancer representations that are resilient to domain shifts. We propose MorphGen (Morphology-Guided Generalization), a method that integrates histopathology images, augmentations, and nuclear segmentation masks within a supervised contrastive learning framework. By aligning latent representations of images and nuclear masks, MorphGen prioritizes diagnostic features such as nuclear and morphological atypia and spatial organization over staining artifacts and domain-specific features. To further enhance out-of-distribution robustness, we incorporate stochastic weight averaging (SWA), steering optimization toward flatter minima. Attention map analyses revealed that MorphGen primarily relies on nuclear morphology, cellular composition, and spatial cell organization within tumors or normal regions for final classification. Finally, we demonstrate resilience of the learned representations to image corruptions (such as staining artifacts) and adversarial attacks, showcasing not only OOD generalization but also addressing critical vulnerabilities in current deep learning systems for digital pathology. Code, datasets, and trained models are available at: https://github.com/hikmatkhan/MorphGen

摘要: 计算组织病理学中的领域概括受到整个切片图像（WSIs）的不均匀性的阻碍，该不均匀性是由机构之间的组织准备、染色和成像条件的变化引起的。与机器学习系统不同，病理学家依赖域不变的形态线索，例如核分裂（增大、轮廓不规则、染色质纹理、空间混乱）、结构分裂（异常结构和腺体形成）以及在不同环境中保持诊断性的总体形态分裂。出于此动机，我们假设，对生物学稳健的核形态和空间组织进行显式建模将能够学习对域转移有弹性的癌症表示。我们提出了MorphGen（形态学引导概括），这是一种将组织病理学图像、增强和核分割掩模集成在监督对比学习框架内的方法。通过对齐图像的潜在表示和核掩模，MorphGen优先考虑核和形态分裂以及空间组织等诊断特征，而不是染色伪影和特定领域的特征。为了进一步增强分布外的鲁棒性，我们引入了随机权重平均（SWA），引导优化走向更平坦的最小值。注意力地图分析表明，MorphGen主要依赖于肿瘤或正常区域内的核形态、细胞组成和空间细胞组织来进行最终分类。最后，我们展示了所学习的表示对图像损坏（例如染色伪影）和对抗攻击的弹性，不仅展示了OOD概括，还解决了当前数字病理学深度学习系统中的关键漏洞。代码、数据集和训练模型可在以下网址获取：https://github.com/hikmatkhan/MorphGen



## **26. A Systematic Approach to Estimate the Security Posture of a Cyber Infrastructure: A Technical Report**

评估网络基础设施安全状况的系统方法：技术报告 cs.CR

11 pages, 5 figures, technical report

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2509.00266v1) [paper-pdf](http://arxiv.org/pdf/2509.00266v1)

**Authors**: Qishen Sam Liang

**Abstract**: Academic and research Cyber Infrastructures (CI) present unique security challenges due to their collaborative nature, heterogeneous components, and the lack of practical, tailored security assessment frameworks. Existing standards can be too generic or complex for CI administrators to apply effectively. This report introduces a systematic, mission-centric approach to estimate and analyze the security posture of a CI. The framework guides administrators through a top-down process: (1) defining unacceptable losses and security missions, (2) identifying associated system hazards and critical assets, and (3) modeling the CI's components and their relationships as a security knowledge graph. The core of this methodology is the construction of directed attack graphs, which systematically map all potential paths an adversary could take from an entry point to a critical asset. By visualizing these attack paths alongside defense mechanisms, the framework provides a clear, comprehensive overview of the system's vulnerabilities and security gaps. This structured approach enables CI operators to proactively assess risks, prioritize mitigation strategies, and make informed, actionable decisions to strengthen the overall security posture of the CI.

摘要: 学术和研究网络基础设施（CI）由于其协作性质、异类组件以及缺乏实用的、量身定制的安全评估框架，带来了独特的安全挑战。现有标准可能过于通用或复杂，CI管理员无法有效应用。本报告介绍了一种系统性、以任务为中心的方法来估计和分析CI的安全态势。该框架指导管理员通过自上而下的流程：（1）定义不可接受的损失和安全任务，（2）识别相关的系统危险和关键资产，以及（3）将CI的组件及其关系建模为安全知识图。该方法的核心是构建定向攻击图，该图系统地映射对手从入口点到关键资产可能采取的所有潜在路径。通过可视化这些攻击路径以及防御机制，该框架提供了系统漏洞和安全漏洞的清晰、全面的概述。这种结构化方法使CI运营商能够主动评估风险、优先考虑缓解策略，并做出明智、可操作的决策，以加强CI的整体安全态势。



## **27. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

检测人工智能代码生成器中的隐形数据中毒攻击 cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.

摘要: 用于自然语言到代码生成的深度学习（DL）模型已成为现代软件开发管道的组成部分。然而，它们严重依赖大量数据，这些数据通常是从未经清理的在线来源收集的，这使它们面临数据中毒攻击，对手会注入恶意样本以微妙地偏向模型行为。最近的有针对性的攻击以语义等效但脆弱的实现悄然取代安全代码，而不依赖显式触发器来发起攻击，这使得检测方法特别难以区分干净样本和有毒样本。我们对这种隐形威胁模型下现有中毒检测方法的有效性进行了系统研究。具体来说，我们对三种DL模型（CodeBRT、CodeT 5+、AST-T5）执行有针对性的中毒，并评估光谱特征分析、激活集群和静态分析作为防御措施。我们的结果表明，所有方法都很难检测无指示器中毒，基于表示的方法无法隔离中毒样本，静态分析会出现假阳性和假阴性，这凸显了人工智能辅助代码生成需要更强大、独立于指示器的防御。



## **28. Adversarial Patch Attack for Ship Detection via Localized Augmentation**

通过局部增强进行船舶检测的对抗补丁攻击 cs.CV

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21472v1) [paper-pdf](http://arxiv.org/pdf/2508.21472v1)

**Authors**: Chun Liu, Panpan Ding, Zheng Zheng, Hailong Wang, Bingqian Zhu, Tao Xu, Zhigang Han, Jiayao Wang

**Abstract**: Current ship detection techniques based on remote sensing imagery primarily rely on the object detection capabilities of deep neural networks (DNNs). However, DNNs are vulnerable to adversarial patch attacks, which can lead to misclassification by the detection model or complete evasion of the targets. Numerous studies have demonstrated that data transformation-based methods can improve the transferability of adversarial examples. However, excessive augmentation of image backgrounds or irrelevant regions may introduce unnecessary interference, resulting in false detections of the object detection model. These errors are not caused by the adversarial patches themselves but rather by the over-augmentation of background and non-target areas. This paper proposes a localized augmentation method that applies augmentation only to the target regions, avoiding any influence on non-target areas. By reducing background interference, this approach enables the loss function to focus more directly on the impact of the adversarial patch on the detection model, thereby improving the attack success rate. Experiments conducted on the HRSC2016 dataset demonstrate that the proposed method effectively increases the success rate of adversarial patch attacks and enhances their transferability.

摘要: 当前基于遥感图像的船舶检测技术主要依赖于深度神经网络（DNN）的物体检测能力。然而，DNN很容易受到对抗补丁攻击，这可能会导致检测模型的错误分类或完全逃避目标。许多研究表明，基于数据转换的方法可以提高对抗性示例的可移植性。然而，图像背景或不相关区域的过度增强可能会引入不必要的干扰，导致对象检测模型的错误检测。这些错误不是由对抗补丁本身引起的，而是由背景和非目标区域的过度扩大引起的。本文提出了一种局部增强方法，仅对目标区域进行增强，避免对非目标区域产生任何影响。通过减少背景干扰，这种方法使损失函数能够更直接地关注对抗补丁对检测模型的影响，从而提高攻击成功率。在HRSC 2016数据集上进行的实验表明，该方法有效提高了对抗性补丁攻击的成功率，增强了其可移植性。



## **29. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

发布到Perish：对LLM辅助同行评审的即时注入攻击 cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.

摘要: 大型语言模型（LLM）越来越多地融入到科学同行评审过程中，这引发了有关其可靠性和操纵弹性的新问题。在这项工作中，我们调查了隐藏的提示注入攻击的可能性，即作者在论文的PDF中嵌入对抗性文本以影响LLM生成的评论。我们首先正式化三种不同的威胁模型，这些模型设想攻击者具有不同的动机--并非所有这些都暗示着恶意意图。对于每个威胁模型，我们设计了对抗性提示，这些提示对人类读者来说是不可见的，但可以将LLM的输出引导到作者想要的结果。通过对领域学者的用户研究，我们得出了四个代表性的审查提示，用于从法学硕士那里获得同行审查。然后，我们评估对抗提示在（i）不同的审查提示、（ii）不同的基于LLM的商业系统和（iii）不同的同行评审论文中的稳健性。我们的结果表明，对抗性提示可以可靠地误导LLM，有时会对“诚实但懒惰”的评论者产生不利影响。最后，我们提出并根据经验评估了在自动内容检查下降低对抗性提示可检测性的方法。



## **30. Time Tells All: Deanonymization of Blockchain RPC Users with Zero Transaction Fee (Extended Version)**

时间证明一切：以零交易费实现区块链PCC用户去匿名化（扩展版本） cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21440v1) [paper-pdf](http://arxiv.org/pdf/2508.21440v1)

**Authors**: Shan Wang, Ming Yang, Yu Liu, Yue Zhang, Shuaiqing Zhang, Zhen Ling, Jiannong Cao, Xinwen Fu

**Abstract**: Remote Procedure Call (RPC) services have become a primary gateway for users to access public blockchains. While they offer significant convenience, RPC services also introduce critical privacy challenges that remain insufficiently examined. Existing deanonymization attacks either do not apply to blockchain RPC users or incur costs like transaction fees assuming an active network eavesdropper. In this paper, we propose a novel deanonymization attack that can link an IP address of a RPC user to this user's blockchain pseudonym. Our analysis reveals a temporal correlation between the timestamps of transaction confirmations recorded on the public ledger and those of TCP packets sent by the victim when querying transaction status. We assume a strong passive adversary with access to network infrastructure, capable of monitoring traffic at network border routers or Internet exchange points. By monitoring network traffic and analyzing public ledgers, the attacker can link the IP address of the TCP packet to the pseudonym of the transaction initiator by exploiting the temporal correlation. This deanonymization attack incurs zero transaction fee. We mathematically model and analyze the attack method, perform large-scale measurements of blockchain ledgers, and conduct real-world attacks to validate the attack. Our attack achieves a high success rate of over 95% against normal RPC users on various blockchain networks, including Ethereum, Bitcoin and Solana.

摘要: 远程过程调用（PRC）服务已成为用户访问公共区块链的主要门户。虽然它们提供了巨大的便利，但它们也带来了关键的隐私挑战，这些挑战仍然没有得到充分的审查。现有的去匿名攻击要么不适用于区块链PRC用户，要么假设有活跃的网络窃听者会产生交易费等成本。在本文中，我们提出了一种新型的去匿名攻击，可以将PRC用户的IP地址链接到该用户的区块链假名。我们的分析揭示了公共分类帐上记录的交易确认的时间戳与受害者在查询交易状态时发送的MAC数据包的时间戳之间存在时间相关性。我们假设一个强大的被动对手，可以访问网络基础设施，能够监控网络边界路由器或互联网交换点的流量。通过监控网络流量和分析公共分类帐，攻击者可以利用时间相关性将Tcp数据包的IP地址链接到交易发起者的假名。这种去匿名化攻击的交易费用为零。我们对攻击方法进行数学建模和分析，对区块链账本进行大规模测量，并进行现实世界的攻击以验证攻击。我们的攻击针对各种区块链网络（包括以太坊、比特币和Solana）上的普通PRC用户，成功率超过95%。



## **31. A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See**

一个全新的世界：创建一个只有人工智能代理才能看到的走廊中毒网络 cs.CR

10 pages, 1 figure

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2509.00124v1) [paper-pdf](http://arxiv.org/pdf/2509.00124v1)

**Authors**: Shaked Zychlinski

**Abstract**: This paper introduces a novel attack vector that leverages website cloaking techniques to compromise autonomous web-browsing agents powered by Large Language Models (LLMs). As these agents become more prevalent, their unique and often homogenous digital fingerprints - comprising browser attributes, automation framework signatures, and network characteristics - create a new, distinguishable class of web traffic. The attack exploits this fingerprintability. A malicious website can identify an incoming request as originating from an AI agent and dynamically serve a different, "cloaked" version of its content. While human users see a benign webpage, the agent is presented with a visually identical page embedded with hidden, malicious instructions, such as indirect prompt injections. This mechanism allows adversaries to hijack agent behavior, leading to data exfiltration, malware execution, or misinformation propagation, all while remaining completely invisible to human users and conventional security crawlers. This work formalizes the threat model, details the mechanics of agent fingerprinting and cloaking, and discusses the profound security implications for the future of agentic AI, highlighting the urgent need for robust defenses against this stealthy and scalable attack.

摘要: 本文介绍了一种新的攻击向量，利用网站伪装技术，以妥协自主的Web浏览代理大语言模型（LLM）。随着这些代理变得越来越普遍，其独特且通常同质的数字指纹-包括浏览器属性，自动化框架签名和网络特征-创建了一个新的，可区分的Web流量类别。攻击利用了这种指纹识别能力。恶意网站可以将传入的请求识别为来自AI代理，并动态提供其内容的不同“隐藏”版本。当人类用户看到良性网页时，代理会呈现一个视觉上相同的页面，其中嵌入了隐藏的恶意指令，例如间接提示注入。该机制允许对手劫持代理行为，导致数据泄露、恶意软件执行或错误信息传播，同时人类用户和传统安全爬虫仍然完全不可见。这项工作正式化了威胁模型，详细介绍了代理指纹识别和隐身的机制，并讨论了代理人工智能未来的深刻安全影响，强调了针对这种隐形和可扩展攻击的强大防御的迫切需要。



## **32. On the Adversarial Robustness of Spiking Neural Networks Trained by Local Learning**

局部学习训练的尖峰神经网络的对抗鲁棒性 cs.LG

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2504.08897v2) [paper-pdf](http://arxiv.org/pdf/2504.08897v2)

**Authors**: Jiaqi Lin, Abhronil Sengupta

**Abstract**: Recent research has shown the vulnerability of Spiking Neural Networks (SNNs) under adversarial examples that are nearly indistinguishable from clean data in the context of frame-based and event-based information. The majority of these studies are constrained in generating adversarial examples using Backpropagation Through Time (BPTT), a gradient-based method which lacks biological plausibility. In contrast, local learning methods, which relax many of BPTT's constraints, remain under-explored in the context of adversarial attacks. To address this problem, we examine adversarial robustness in SNNs through the framework of four types of training algorithms. We provide an in-depth analysis of the ineffectiveness of gradient-based adversarial attacks to generate adversarial instances in this scenario. To overcome these limitations, we introduce a hybrid adversarial attack paradigm that leverages the transferability of adversarial instances. The proposed hybrid approach demonstrates superior performance, outperforming existing adversarial attack methods. Furthermore, the generalizability of the method is assessed under multi-step adversarial attacks, adversarial attacks in black-box FGSM scenarios, and within the non-spiking domain.

摘要: 最近的研究表明，尖峰神经网络（SNN）在对抗性示例下的脆弱性，这些示例与基于帧和基于事件的信息背景下的干净数据几乎无法区分。这些研究中的大多数在使用时间反向传播（BPTT）生成对抗性示例方面受到限制，BPTT是一种基于梯度的方法，缺乏生物学相似性。相比之下，本地学习方法放松了BPTT的许多限制，但在对抗性攻击的背景下仍然没有得到充分的探索。为了解决这个问题，我们通过四种类型的训练算法的框架来检查SNN中的对抗鲁棒性。我们深入分析了基于梯度的对抗攻击在这种情况下生成对抗实例的无效性。为了克服这些限制，我们引入了一种混合对抗攻击范式，该范式利用对抗实例的可移植性。提出的混合方法表现出卓越的性能，优于现有的对抗攻击方法。此外，在多步对抗性攻击、黑匣子FGSM场景中的对抗性攻击和非尖峰域中评估了该方法的通用性。



## **33. The WASM Cloak: Evaluating Browser Fingerprinting Defenses Under WebAssembly based Obfuscation**

WASM斗篷：在基于WebAssembly的混淆下评估浏览器指纹防御 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21219v1) [paper-pdf](http://arxiv.org/pdf/2508.21219v1)

**Authors**: A H M Nazmus Sakib, Mahsin Bin Akram, Joseph Spracklen, Sahan Kalutarage, Raveen Wijewickrama, Igor Bilogrevic, Murtuza Jadliwala

**Abstract**: Browser fingerprinting defenses have historically focused on detecting JavaScript(JS)-based tracking techniques. However, the widespread adoption of WebAssembly (WASM) introduces a potential blind spot, as adversaries can convert JS to WASM's low-level binary format to obfuscate malicious logic. This paper presents the first systematic evaluation of how such WASM-based obfuscation impacts the robustness of modern fingerprinting defenses. We develop an automated pipeline that translates real-world JS fingerprinting scripts into functional WASM-obfuscated variants and test them against two classes of defenses: state-of-the-art detectors in research literature and commercial, in-browser tools. Our findings reveal a notable divergence: detectors proposed in the research literature that rely on feature-based analysis of source code show moderate vulnerability, stemming from outdated datasets or a lack of WASM compatibility. In contrast, defenses such as browser extensions and native browser features remained completely effective, as their API-level interception is agnostic to the script's underlying implementation. These results highlight a gap between academic and practical defense strategies and offer insights into strengthening detection approaches against WASM-based obfuscation, while also revealing opportunities for more evasive techniques in future attacks.

摘要: 浏览器指纹识别防御历来专注于检测基于JavaScript（JS）的跟踪技术。然而，WebAssembly（WASM）的广泛采用引入了一个潜在的盲点，因为对手可以将JS转换为WASM的低级二进制格式来混淆恶意逻辑。本文首次对这种基于WASM的模糊处理如何影响现代指纹防御的稳健性进行了系统评估。我们开发了一个自动化管道，将现实世界的JS指纹识别脚本翻译为功能性WASM模糊变体，并针对两类防御措施对其进行测试：研究文献中最先进的检测器和商业浏览器内工具。我们的研究结果揭示了一个显着的分歧：研究文献中提出的依赖于源代码基于特征的分析的检测器显示出中度漏洞，源于过时的数据集或缺乏WASM兼容性。相比之下，浏览器扩展和原生浏览器功能等防御仍然完全有效，因为它们的API级拦截与脚本的底层实现无关。这些结果凸显了学术和实际防御策略之间的差距，并为加强针对基于WASM的模糊的检测方法提供了见解，同时也揭示了未来攻击中使用更具规避性技术的机会。



## **34. First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge**

NeurIPS 2024隐形水印去除挑战赛的一流解决方案 cs.CV

Winning solution to the NeurIPS 2024 Erasing the Invisible challenge

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21072v1) [paper-pdf](http://arxiv.org/pdf/2508.21072v1)

**Authors**: Fahad Shamshad, Tameem Bakr, Yahia Shaaban, Noor Hussein, Karthik Nandakumar, Nils Lukas

**Abstract**: Content watermarking is an important tool for the authentication and copyright protection of digital media. However, it is unclear whether existing watermarks are robust against adversarial attacks. We present the winning solution to the NeurIPS 2024 Erasing the Invisible challenge, which stress-tests watermark robustness under varying degrees of adversary knowledge. The challenge consisted of two tracks: a black-box and beige-box track, depending on whether the adversary knows which watermarking method was used by the provider. For the beige-box track, we leverage an adaptive VAE-based evasion attack, with a test-time optimization and color-contrast restoration in CIELAB space to preserve the image's quality. For the black-box track, we first cluster images based on their artifacts in the spatial or frequency-domain. Then, we apply image-to-image diffusion models with controlled noise injection and semantic priors from ChatGPT-generated captions to each cluster with optimized parameter settings. Empirical evaluations demonstrate that our method successfully achieves near-perfect watermark removal (95.7%) with negligible impact on the residual image's quality. We hope that our attacks inspire the development of more robust image watermarking methods.

摘要: 内容水印是数字媒体认证和版权保护的重要工具。然而，目前尚不清楚现有的水印是否能够抵御对抗攻击。我们向NeurIPS 2024 Erase the Invisible挑战展示了获胜的解决方案，该解决方案在不同程度的对手知识下对水印稳健性进行压力测试。挑战由两个轨道组成：黑匣子和米色盒轨道，具体取决于对手是否知道提供商使用了哪种水印方法。对于米色盒轨道，我们利用基于VAR的自适应规避攻击，并在CIELAB空间中进行测试时优化和颜色对比度恢复，以保持图像的质量。对于黑匣子轨道，我们首先根据图像在空间或频域中的伪影对图像进行分组。然后，我们将具有受控噪音注入和ChatGPT生成的字幕的语义先验的图像到图像扩散模型应用到每个集群，并优化参数设置。经验评估表明，我们的方法成功实现了近乎完美的水印去除（95.7%），而对残留图像质量的影响可以忽略不计。我们希望我们的攻击能够激发更鲁棒的图像水印方法的开发。



## **35. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **36. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

Token Buncher：保护LLM免受有害的强化学习微调 cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.

摘要: 随着大型语言模型（LLM）的能力不断增强，通过微调导致有害误用的风险也在增加。虽然大多数先前的研究都假设攻击者依赖监督微调（SFT）来进行此类滥用，但我们系统地证明，强化学习（RL）使对手能够在匹配的计算预算下更有效地打破安全对齐并促进高级有害任务协助。为了应对这种新出现的威胁，我们提出了TokenBuncher，这是第一个专门针对基于RL的有害微调的有效防御。TokenBuncher抑制了RL所依赖的基础：模型响应不确定性。通过限制不确定性，基于RL的微调无法再利用不同的奖励信号来推动模型走向有害行为。我们通过以互赏为回报的RL和旨在防止专家域有害能力升级的Token Noiser机制来实现这种防御。跨多个模型和RL算法的大量实验表明，TokenBuncher稳健地减轻了有害的RL微调，同时保留了良性的任务效用和微调能力。我们的结果强调，基于RL的有害微调比SFT构成更大的系统性风险，并且TokenBuncher提供了有效且通用的防御。



## **37. Disruptive Attacks on Face Swapping via Low-Frequency Perceptual Perturbations**

通过低频感知扰动对人脸交换进行破坏性攻击 cs.CV

Accepted to IEEE IJCNN 2025

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20595v1) [paper-pdf](http://arxiv.org/pdf/2508.20595v1)

**Authors**: Mengxiao Huang, Minglei Shu, Shuwang Zhou, Zhaoyang Liu

**Abstract**: Deepfake technology, driven by Generative Adversarial Networks (GANs), poses significant risks to privacy and societal security. Existing detection methods are predominantly passive, focusing on post-event analysis without preventing attacks. To address this, we propose an active defense method based on low-frequency perceptual perturbations to disrupt face swapping manipulation, reducing the performance and naturalness of generated content. Unlike prior approaches that used low-frequency perturbations to impact classification accuracy,our method directly targets the generative process of deepfake techniques. We combine frequency and spatial domain features to strengthen defenses. By introducing artifacts through low-frequency perturbations while preserving high-frequency details, we ensure the output remains visually plausible. Additionally, we design a complete architecture featuring an encoder, a perturbation generator, and a decoder, leveraging discrete wavelet transform (DWT) to extract low-frequency components and generate perturbations that disrupt facial manipulation models. Experiments on CelebA-HQ and LFW demonstrate significant reductions in face-swapping effectiveness, improved defense success rates, and preservation of visual quality.

摘要: 由生成对抗网络（GAN）驱动的Deepfake技术对隐私和社会安全构成了重大风险。现有的检测方法主要是被动的，重点是事后分析，而不防止攻击。为了解决这个问题，我们提出了一种基于低频感知扰动的主动防御方法，以扰乱面部交换操纵，降低生成内容的性能和自然性。与使用低频扰动影响分类准确性的现有方法不同，我们的方法直接针对深度伪造技术的生成过程。我们结合频率和空间域特征来加强防御。通过通过低频扰动引入伪影，同时保留高频细节，我们确保输出在视觉上保持可信。此外，我们设计了一个完整的架构，其中包括编码器、扰动生成器和解码器，利用离散子波变换（DWT）来提取低频分量并生成扰乱面部操纵模型的扰动。CelebA-HQ和LFW的实验表明，换脸有效性显着降低，防御成功率提高，视觉质量得以保留。



## **38. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

多模态LLM越狱的概率建模：从量化到应用 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 最近，多模式大型语言模型（MLLM）展示了其在理解多模式内容方面的卓越能力。然而，它们仍然容易受到越狱攻击，这些攻击利用其安全调整中的弱点来产生有害反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLM的能力的二元分类是不合适的。从这个观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表当提示此输入时MLLM生成恶意响应的可能性。我们通过对MLLM的多次查询来估算这一可能性。使用越狱概率预测网络（JPPN）对输入隐藏状态与其相应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体来说，我们提出了基于越狱概率的攻击（JPA），该攻击优化输入图像上的对抗性扰动以最大化越狱概率，并通过包括单调文本改写进一步将其增强为多模式JPA（MJPA）。为了对抗攻击，我们还提出了基于越狱概率的微调（JPF），它通过MLLM参数更新最大限度地降低越狱概率。大量实验表明，（1）（M）JPA在白盒和黑匣子设置下攻击广泛的模型时都能产生显着的改进。(2)JPF最多将越狱人数大幅减少60%以上。上述两个结果都证明了引入越狱概率以在输入越狱能力之间进行细微差别的重要性。



## **39. Formal Verification of Physical Layer Security Protocols for Next-Generation Communication Networks (extended version)**

下一代通信网络物理层安全协议的形式验证（扩展版本） cs.CR

Extended version (with appendices) of the camera-ready for ICFEM2025;  24 pages, 3 tables, and 6 figures

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.19430v2) [paper-pdf](http://arxiv.org/pdf/2508.19430v2)

**Authors**: Kangfeng Ye, Roberto Metere, Jim Woodcock, Poonam Yadav

**Abstract**: Formal verification is crucial for ensuring the robustness of security protocols against adversarial attacks. The Needham-Schroeder protocol, a foundational authentication mechanism, has been extensively studied, including its integration with Physical Layer Security (PLS) techniques such as watermarking and jamming. Recent research has used ProVerif to verify these mechanisms in terms of secrecy. However, the ProVerif-based approach limits the ability to improve understanding of security beyond verification results. To overcome these limitations, we re-model the same protocol using an Isabelle formalism that generates sound animation, enabling interactive and automated formal verification of security protocols. Our modelling and verification framework is generic and highly configurable, supporting both cryptography and PLS. For the same protocol, we have conducted a comprehensive analysis (secrecy and authenticity in four different eavesdropper locations under both passive and active attacks) using our new web interface. Our findings not only successfully reproduce and reinforce previous results on secrecy but also reveal an uncommon but expected outcome: authenticity is preserved across all examined scenarios, even in cases where secrecy is compromised. We have proposed a PLS-based Diffie-Hellman protocol that integrates watermarking and jamming, and our analysis shows that it is secure for deriving a session key with required authentication. These highlight the advantages of our novel approach, demonstrating its robustness in formally verifying security properties beyond conventional methods.

摘要: 形式验证对于确保安全协议抵御对抗攻击的稳健性至关重要。Needham-Schroeder协议是一种基础认证机制，已被广泛研究，包括其与水印和干扰等物理层安全（SCS）技术的集成。最近的研究使用ProVerif来验证这些机制的保密性。然而，基于ProVerif的方法限制了提高对验证结果之外的安全性理解的能力。为了克服这些限制，我们使用Isabelle形式主义重新建模相同的协议，该形式主义生成声音动画，从而实现安全协议的交互式和自动化形式验证。我们的建模和验证框架是通用的且高度可配置的，支持加密技术和最大限度地支持。对于同一协议，我们使用新的网络界面进行了全面的分析（被动和主动攻击下四个不同窃听者位置的保密性和真实性）。我们的发现不仅成功地复制和强化了之前的保密结果，而且揭示了一个不寻常但预期的结果：在所有检查的场景中，真实性都得到了保留，即使在保密性受到损害的情况下。我们提出了一种基于PL的迪夫-赫尔曼协议，集成了水印和干扰，我们的分析表明，它对于推导具有所需认证的会话密钥是安全的。这些凸显了我们的新型方法的优势，证明了其在正式验证安全属性方面的鲁棒性，超出了传统方法。



## **40. Enhancing Resilience for IoE: A Perspective of Networking-Level Safeguard**

增强IoE的弹性：网络级保障的视角 cs.CR

To be published in IEEE Network Magazine, 2026

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20504v1) [paper-pdf](http://arxiv.org/pdf/2508.20504v1)

**Authors**: Guan-Yan Yang, Jui-Ning Chen, Farn Wang, Kuo-Hui Yeh

**Abstract**: The Internet of Energy (IoE) integrates IoT-driven digital communication with power grids to enable efficient and sustainable energy systems. Still, its interconnectivity exposes critical infrastructure to sophisticated cyber threats, including adversarial attacks designed to bypass traditional safeguards. Unlike general IoT risks, IoE threats have heightened public safety consequences, demanding resilient solutions. From the networking-level safeguard perspective, we propose a Graph Structure Learning (GSL)-based safeguards framework that jointly optimizes graph topology and node representations to resist adversarial network model manipulation inherently. Through a conceptual overview, architectural discussion, and case study on a security dataset, we demonstrate GSL's superior robustness over representative methods, offering practitioners a viable path to secure IoE networks against evolving attacks. This work highlights the potential of GSL to enhance the resilience and reliability of future IoE networks for practitioners managing critical infrastructure. Lastly, we identify key open challenges and propose future research directions in this novel research area.

摘要: 能源互联网（IoE）将物联网驱动的数字通信与电网集成，以实现高效和可持续的能源系统。尽管如此，其互连性仍使关键基础设施面临复杂的网络威胁，包括旨在绕过传统保障措施的对抗性攻击。与一般的物联网风险不同，IoE威胁加剧了公共安全后果，需要弹性解决方案。从网络级保障的角度，我们提出了一个基于图结构学习（GSL）的保障框架，该框架联合优化图布局和节点表示，以本质上抵抗对抗性网络模型操纵。通过对安全数据集的概念概述、架构讨论和案例研究，我们展示了GSL优于代表性方法的卓越稳健性，为从业者提供了一条保护IoE网络免受不断发展的攻击的可行途径。这项工作强调了GSL为管理关键基础设施的从业者增强未来IoE网络的弹性和可靠性的潜力。最后，我们确定了关键的开放挑战，并提出了这个新颖研究领域的未来研究方向。



## **41. When Memory Becomes a Vulnerability: Towards Multi-turn Jailbreak Attacks against Text-to-Image Generation Systems**

当内存成为漏洞时：针对文本到图像生成系统的多回合越狱攻击 cs.CV

This work proposes a multi-turn jailbreak attack against real-world  chat-based T2I generation systems that intergrate memory mechanism. It also  constructed a simulation system, with considering three industrial-grade  memory mechanisms, 7 kinds of safety filters (both input and output)

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2504.20376v2) [paper-pdf](http://arxiv.org/pdf/2504.20376v2)

**Authors**: Shiqian Zhao, Jiayang Liu, Yiming Li, Runyi Hu, Xiaojun Jia, Wenshu Fan, Xinfeng Li, Jie Zhang, Wei Dong, Tianwei Zhang, Luu Anh Tuan

**Abstract**: Modern text-to-image (T2I) generation systems (e.g., DALL$\cdot$E 3) exploit the memory mechanism, which captures key information in multi-turn interactions for faithful generation. Despite its practicality, the security analyses of this mechanism have fallen far behind. In this paper, we reveal that it can exacerbate the risk of jailbreak attacks. Previous attacks fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or lead to the generation of non-unsafe images due to under- or over-detoxification. In contrast, we propose embedding the malice at the inception of the chat session in memory, addressing the above limitations.   Specifically, we propose Inception, the first multi-turn jailbreak attack against real-world text-to-image generation systems that explicitly exploits their memory mechanisms. Inception is composed of two key modules: segmentation and recursion. We introduce Segmentation, a semantic-preserving method that generates multi-round prompts. By leveraging NLP analysis techniques, we design policies to decompose a prompt, together with its malicious intent, according to sentence structure, thereby evading safety filters. Recursion further addresses the challenge posed by unsafe sub-prompts that cannot be separated through simple segmentation. It firstly expands the sub-prompt, then invokes segmentation recursively. To facilitate multi-turn adversarial prompts crafting, we build VisionFlow, an emulation T2I system that integrates two-stage safety filters and industrial-grade memory mechanisms. The experiment results show that Inception successfully allures unsafe image generation, surpassing the SOTA by a 20.0\% margin in attack success rate. We also conduct experiments on the real-world commercial T2I generation platforms, further validating the threats of Inception in practice.

摘要: 现代文本到图像（T2 I）生成系统（例如，DALL$\csot $E 3）利用记忆机制，该机制捕获多回合交互中的关键信息，以供忠实生成。尽管该机制具有实用性，但对该机制的安全分析却远远落后。在本文中，我们揭示了它会加剧越狱攻击的风险。之前的攻击将不安全的目标提示融合为一个最终的对抗提示，可以很容易地检测到这一点，或者由于解毒不足或过度而导致生成不安全的图像。相比之下，我们建议将聊天会话开始时的恶意嵌入到内存中，以解决上述限制。   具体来说，我们提出了Incept，这是第一个针对现实世界的文本到图像生成系统的多回合越狱攻击，该系统明确利用了其记忆机制。Incion由两个关键模块组成：分段和回归。我们引入了Segmentation，这是一种生成多轮提示的语义保留方法。通过利用NLP分析技术，我们设计策略根据句子结构分解提示及其恶意意图，从而规避安全过滤器。回归进一步解决了无法通过简单分段分离的不安全子提示所带来的挑战。它首先扩展子提示，然后循环调用分段。为了促进多回合对抗提示的制作，我们构建了VisionFlow，这是一个集成两级安全过滤器和工业级存储机制的仿真T2 I系统。实验结果表明，Incion成功诱导了不安全的图像生成，攻击成功率超过SOTA 20.0%。我们还在现实世界的商业T2 I一代平台上进行了实验，进一步验证了Incept在实践中的威胁。



## **42. Safe-Control: A Safety Patch for Mitigating Unsafe Content in Text-to-Image Generation Models**

Safe-Control：用于缓解文本到图像生成模型中不安全内容的安全补丁 cs.CV

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21099v1) [paper-pdf](http://arxiv.org/pdf/2508.21099v1)

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo

**Abstract**: Despite the advancements in Text-to-Image (T2I) generation models, their potential for misuse or even abuse raises serious safety concerns. Model developers have made tremendous efforts to introduce safety mechanisms that can address these concerns in T2I models. However, the existing safety mechanisms, whether external or internal, either remain susceptible to evasion under distribution shifts or require extensive model-specific adjustments. To address these limitations, we introduce Safe-Control, an innovative plug-and-play safety patch designed to mitigate unsafe content generation in T2I models. Using data-driven strategies and safety-aware conditions, Safe-Control injects safety control signals into the locked T2I model, acting as an update in a patch-like manner. Model developers can also construct various safety patches to meet the evolving safety requirements, which can be flexibly merged into a single, unified patch. Its plug-and-play design further ensures adaptability, making it compatible with other T2I models of similar denoising architecture. We conduct extensive evaluations on six diverse and public T2I models. Empirical results highlight that Safe-Control is effective in reducing unsafe content generation across six diverse T2I models with similar generative architectures, yet it successfully maintains the quality and text alignment of benign images. Compared to seven state-of-the-art safety mechanisms, including both external and internal defenses, Safe-Control significantly outperforms all baselines in reducing unsafe content generation. For example, it reduces the probability of unsafe content generation to 7%, compared to approximately 20% for most baseline methods, under both unsafe prompts and the latest adversarial attacks.

摘要: 尽管文本到图像（T2 I）生成模型取得了进步，但它们被滥用甚至滥用的可能性引发了严重的安全问题。模型开发人员做出了巨大努力来引入可以解决T2 I模型中这些问题的安全机制。然而，现有的安全机制，无论是外部的还是内部的，要么仍然容易在分配转移下被规避，要么需要针对特定模型的广泛调整。为了解决这些限制，我们引入了Safe-Control，这是一种创新的即插即用安全补丁，旨在减轻T2 I模型中的不安全内容生成。Safe-Control使用数据驱动策略和安全意识条件，将安全控制信号注入锁定的T2 I模型，以类似补丁的方式充当更新。模型开发人员还可以构建各种安全补丁来满足不断变化的安全要求，这些补丁可以灵活地合并到单个、统一的补丁中。其即插即用设计进一步确保了适应性，使其与类似去噪架构的其他T2 I型号兼容。我们对六种多样化的公共T2 I模型进行了广泛的评估。经验结果强调，Safe-Control可以有效减少具有相似生成架构的六种不同T2 I模型中的不安全内容生成，但它成功地保持了良性图像的质量和文本对齐。与七种最先进的安全机制（包括外部和内部防御）相比，Safe-Control在减少不安全内容生成方面显着优于所有基线。例如，在不安全提示和最新的对抗性攻击下，它将不安全内容生成的可能性降低到7%，而大多数基线方法的可能性约为20%。



## **43. Enhanced Rényi Entropy-Based Post-Quantum Key Agreement with Provable Security and Information-Theoretic Guarantees**

增强的基于Rényi Entropy的后量子密钥协议，具有可证明的安全性和信息理论保证 cs.CR

11 pages, 3 tables

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2509.00104v1) [paper-pdf](http://arxiv.org/pdf/2509.00104v1)

**Authors**: Ruopengyu Xu, Chenglian Liu

**Abstract**: This paper presents an enhanced post-quantum key agreement protocol based on R\'{e}nyi entropy, addressing vulnerabilities in the original construction while preserving information-theoretic security properties. We develop a theoretical framework leveraging entropy-preserving operations and secret-shared verification to achieve provable security against quantum adversaries. Through entropy amplification techniques and quantum-resistant commitments, the protocol establishes $2^{128}$ quantum security guarantees under the quantum random oracle model. Key innovations include a confidentiality-preserving verification mechanism using distributed polynomial commitments, tightened min-entropy bounds with guaranteed non-negativity, and composable security proofs in the quantum universal composability framework. Unlike computational approaches, our method provides information-theoretic security without hardness assumptions while maintaining polynomial complexity. Theoretical analysis demonstrates resilience against known quantum attack vectors, including Grover-accelerated brute force and quantum memory attacks. The protocol achieves parameterization for 128-bit quantum security with efficient $\mathcal{O}(n^2)$ communication complexity. Extensions to secure multiparty computation and quantum network applications are established, providing a foundation for long-term cryptographic security. All security claims are derived from mathematical proofs; this theoretical work presents no experimental validation.

摘要: 本文提出了一种基于R\'{e}nyi信息的增强型后量子密钥协商协议，解决了原始结构中的漏洞，同时保留了信息论安全属性。我们开发了一个理论框架，利用保留信息的操作和秘密共享验证来实现针对量子对手的可证明安全性。通过熵放大技术和量子抵抗承诺，该协议在量子随机预言模型下建立了价值2亿美元的量子安全保证。关键创新包括使用分布式多项承诺的保密性保护验证机制、具有保证非负性的收紧最小熵界限以及量子普适可组合性框架中的可组合安全性证明。与计算方法不同，我们的方法提供了信息理论安全性，无需硬性假设，同时保持了多元复杂性。理论分析证明了对已知量子攻击载体的弹性，包括Grover加速的暴力和量子存储攻击。该协议实现了128位量子安全的参数化，具有高效的$\mathCal{O}（n#2）$通信复杂性。建立了安全多方计算和量子网络应用的扩展，为长期加密安全提供了基础。所有安全主张都源自数学证明;这项理论工作没有进行实验验证。



## **44. Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs**

一次毒药，永远拒绝：重新调整在LLM中注入偏见的对齐 cs.LG

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20333v1) [paper-pdf](http://arxiv.org/pdf/2508.20333v1)

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.

摘要: 大型语言模型（LLM）通过训练它们拒绝回答有害或不安全的提示来满足道德标准和安全要求。在本文中，我们展示了对手如何利用LLM的一致性来植入偏见，或实施有针对性的审查，而不会降低模型对无关主题的响应能力。具体来说，我们提出了颠覆性对齐注入（SAI），这是一种毒害攻击，利用对齐机制来触发对对手预定义的特定主题或查询的拒绝。尽管通过过度对齐引发拒绝可能并不奇怪，但我们展示了如何利用这种拒绝来向模型中注入偏见。令人惊讶的是，SAI回避了最先进的中毒防御，包括LLM状态取证，以及旨在检测FL环境中中毒的强大聚合技术。我们通过说明这种攻击对LLM支持的应用程序管道的端到端影响来证明这种攻击的实际危险。对于ChatDoctor等基于聊天的应用程序来说，存在1%的数据中毒，系统拒绝回答针对目标种族类别的医疗保健问题，导致高度偏见（$\Delta DP$为23%）。我们还表明，在其他NLP任务中也可能会引发偏见：对于一个简历选择管道来说，拒绝汇总来自所选大学的简历，选择中会出现高偏见（$\Delta DP$为27%）。其他9个基于聊天的下游应用程序会产生更高的偏差（$\Delta DP$~38%）。



## **45. Differentially Private Federated Quantum Learning via Quantum Noise**

通过量子噪音的差异化私有联邦量子学习 quant-ph

This paper has been accepted at 2025 IEEE International Conference on  Quantum Computing and Engineering (QCE)

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20310v1) [paper-pdf](http://arxiv.org/pdf/2508.20310v1)

**Authors**: Atit Pokharel, Ratun Rahman, Shaba Shaon, Thomas Morris, Dinh C. Nguyen

**Abstract**: Quantum federated learning (QFL) enables collaborative training of quantum machine learning (QML) models across distributed quantum devices without raw data exchange. However, QFL remains vulnerable to adversarial attacks, where shared QML model updates can be exploited to undermine information privacy. In the context of noisy intermediate-scale quantum (NISQ) devices, a key question arises: How can inherent quantum noise be leveraged to enforce differential privacy (DP) and protect model information during training and communication? This paper explores a novel DP mechanism that harnesses quantum noise to safeguard quantum models throughout the QFL process. By tuning noise variance through measurement shots and depolarizing channel strength, our approach achieves desired DP levels tailored to NISQ constraints. Simulations demonstrate the framework's effectiveness by examining the relationship between differential privacy budget and noise parameters, as well as the trade-off between security and training accuracy. Additionally, we demonstrate the framework's robustness against an adversarial attack designed to compromise model performance using adversarial examples, with evaluations based on critical metrics such as accuracy on adversarial examples, confidence scores for correct predictions, and attack success rates. The results reveal a tunable trade-off between privacy and robustness, providing an efficient solution for secure QFL on NISQ devices with significant potential for reliable quantum computing applications.

摘要: 量子联邦学习（QFL）支持跨分布式量子设备协作训练量子机器学习（QML）模型，无需原始数据交换。然而，QFL仍然容易受到对抗攻击，共享的QML模型更新可能被利用来破坏信息隐私。在有噪音的中等规模量子（NISQ）设备的背景下，出现了一个关键问题：如何利用固有量子噪音来在训练和通信期间实施差异隐私（DP）并保护模型信息？本文探索了一种新型的DP机制，该机制利用量子噪音在整个QFL过程中保护量子模型。通过通过测量镜头和去极化通道强度来调整噪音方差，我们的方法可以实现针对NISQ约束定制的所需DP水平。模拟通过检查差异隐私预算和噪音参数之间的关系以及安全性和训练准确性之间的权衡来证明该框架的有效性。此外，我们还展示了该框架对旨在使用对抗性示例损害模型性能的对抗性攻击的稳健性，评估基于关键指标，例如对抗性示例的准确性、正确预测的置信度分数和攻击成功率。结果揭示了隐私和稳健性之间的可调权衡，为NISQ设备上的安全QFL提供了一种有效的解决方案，具有可靠的量子计算应用的巨大潜力。



## **46. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

CoCoTen：通过上下文共现张量的潜在空间特征检测大型语言模型的对抗性输入 cs.CL

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.02997v3) [paper-pdf](http://arxiv.org/pdf/2508.02997v3)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models.

摘要: 大型语言模型（LLM）在许多应用中的广泛使用标志着研究和实践的重大进步。然而，它们的复杂性和难以理解的性质使它们容易受到攻击，尤其是旨在产生有害反应的越狱。为了应对这些威胁，开发强大的检测方法对于安全可靠地使用LLM至关重要。本文使用上下文共生矩阵来研究这个检测问题，该结构因其在数据稀缺环境中的有效性而被公认。我们提出了一种利用上下文同现矩阵和张量的潜在空间特征的新型方法，以有效识别对抗和越狱提示。我们的评估表明，这种方法仅使用0.5%的标记提示即可获得显着的0.83分，比基线提高了96.6%。这一结果凸显了我们所学习模式的力量，尤其是当标记数据稀缺时。与基线模型相比，我们的方法也明显更快，加速范围为2.3至128.4倍。



## **47. Network-Level Prompt and Trait Leakage in Local Research Agents**

本地研究代理的网络级提示和特征泄露 cs.CR

under review

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20282v1) [paper-pdf](http://arxiv.org/pdf/2508.20282v1)

**Authors**: Hyejun Jeong, Mohammadreze Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Badasarian

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed \emph{locally} by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.   Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73\% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29\%.

摘要: 我们表明，Web和研究代理（WRA）--调查互联网上复杂主题的基于语言模型的系统--很容易受到ISP等被动网络对手的推理攻击。这些代理可以由组织和个人出于隐私、法律或财务目的\{本地}部署。与人类零星的网络浏览不同，WRA访问具有可区分的时间相关性的价值70 {-}140美元的域名，从而实现独特的指纹攻击。   具体来说，我们展示了针对WRA的新型提示和用户特征泄露攻击，该攻击仅利用其网络级元数据（即，访问的IP地址及其时间）。我们首先根据用户搜索查询和合成人物角色生成的查询构建新的WRA痕迹数据集。我们定义了一个行为指标（称为OBELS）来全面评估原始提示和推断提示之间的相似性，表明我们的攻击恢复了用户提示73%以上的功能和领域知识。扩展到多会话设置，我们可以高准确性地恢复32个潜在特征中的多达19个。我们的攻击在部分可观察性和噪音条件下仍然有效。最后，我们讨论了限制域多样性或混淆痕迹的缓解策略，显示出可忽略的实用性影响，同时将攻击有效性平均降低29%。



## **48. Adversarial Manipulation of Reasoning Models using Internal Representations**

使用内部表示的推理模型的对抗性操纵 cs.CL

Accepted to the ICML 2025 Workshop on Reliable and Responsible  Foundation Models (R2FM). 20 pages, 12 figures

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.03167v2) [paper-pdf](http://arxiv.org/pdf/2507.03167v2)

**Authors**: Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi

**Abstract**: Reasoning models generate chain-of-thought (CoT) tokens before their final output, but how this affects their vulnerability to jailbreak attacks remains unclear. While traditional language models make refusal decisions at the prompt-response boundary, we find evidence that DeepSeek-R1-Distill-Llama-8B makes these decisions within its CoT generation. We identify a linear direction in activation space during CoT token generation that predicts whether the model will refuse or comply -- termed the "caution" direction because it corresponds to cautious reasoning patterns in the generated text. Ablating this direction from model activations increases harmful compliance, effectively jailbreaking the model. We additionally show that intervening only on CoT token activations suffices to control final outputs, and that incorporating this direction into prompt-based attacks improves success rates. Our findings suggest that the chain-of-thought itself is a promising new target for adversarial manipulation in reasoning models. Code available at https://github.com/ky295/reasoning-manipulation.

摘要: 推理模型在最终输出之前生成思想链（CoT）代币，但这如何影响其对越狱攻击的脆弱性尚不清楚。虽然传统的语言模型在拒绝-响应边界上做出拒绝决定，但我们发现有证据表明DeepSeek-R1-Distill-Llama-8B在其CoT生成中做出了这些决定。我们在CoT令牌生成过程中确定了激活空间中的线性方向，该方向预测模型是拒绝还是遵守-称为“谨慎”方向，因为它对应于生成文本中的谨慎推理模式。从模型激活中移除这个方向会增加有害的遵从性，有效地越狱模型。此外，我们还表明，仅干预CoT令牌激活就足以控制最终输出，并且将此方向纳入基于令牌的攻击可以提高成功率。我们的研究结果表明，思维链本身是推理模型中对抗性操纵的一个有前途的新目标。代码可访问https://github.com/ky295/reasoning-manipulation。



## **49. Scaling Decentralized Learning with FLock**

利用Flock扩展去中心化学习 cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.

摘要: 由于集中控制的不足以及分散式方案的大量计算和通信负担，大型语言模型（LLM）的微调受到阻碍。虽然典型的标准联邦学习（FL）支持数据隐私，但中央服务器要求会创建单点攻击和中毒攻击的脆弱性。将这一方向的结果推广到异类、无信任环境中的70 B参数模型已被证明是一个巨大但未突破的瓶颈。本文介绍了Flock，这是一个用于安全高效协作LLM微调的去中心化框架。Flock将基于区块链的信任层与经济激励相结合，用安全、可审计的协议取代了中央聚合器，用于不受信任方之间的合作。我们首次对在安全、多域、去中心化的环境中微调70 B LLM进行了实证验证。我们的实验表明，Flock框架可以抵御后门中毒攻击，这些攻击会损害标准FL优化器并促进协同知识转移。由此产生的模型显示对抗性攻击成功率降低了>68%。全局模型还展示了卓越的跨域泛化能力，优于在自己的专业数据上孤立训练的模型。



## **50. Cell-Free Massive MIMO-Based Physical-Layer Authentication**

无细胞大规模基于MMO的物理层认证 eess.SP

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19931v1) [paper-pdf](http://arxiv.org/pdf/2508.19931v1)

**Authors**: Isabella W. G. da Silva, Zahra Mobini, Hien Quoc Ngo, Michail Matthaiou

**Abstract**: In this paper, we exploit the cell-free massive multiple-input multiple-output (CF-mMIMO) architecture to design a physical-layer authentication (PLA) framework that can simultaneously authenticate multiple distributed users across the coverage area. Our proposed scheme remains effective even in the presence of active adversaries attempting impersonation attacks to disrupt the authentication process. Specifically, we introduce a tag-based PLA CFmMIMO system, wherein the access points (APs) first estimate their channels with the legitimate users during an uplink training phase. Subsequently, a unique secret key is generated and securely shared between each user and the APs. We then formulate a hypothesis testing problem and derive a closed-form expression for the probability of detection for each user in the network. Numerical results validate the effectiveness of the proposed approach, demonstrating that it maintains a high detection probability even as the number of users in the system increases.

摘要: 本文利用无单元大规模多输入多输出（CF-mMMO）架构设计了一个物理层认证（PLA）框架，该框架可以同时对覆盖区域内的多个分布式用户进行认证。即使存在试图模仿攻击以破坏身份验证过程的活跃对手，我们提出的方案仍然有效。具体来说，我们引入了一种基于标签的PLA CFmMMO系统，其中接入点（AP）首先在上行链路训练阶段估计其与合法用户的信道。随后，生成唯一的秘密密钥并在每个用户和AP之间安全共享。然后，我们制定一个假设测试问题，并推导出网络中每个用户的检测概率的封闭形式表达。数值结果验证了所提出方法的有效性，表明即使系统中用户数量增加，它也能保持高检测概率。



