# Latest Adversarial Attack Papers
**update at 2025-03-21 10:41:20**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Graph of Effort: Quantifying Risk of AI Usage for Vulnerability Assessment**

努力图表：量化人工智能使用风险以进行漏洞评估 cs.CR

8 pages; accepted for the 16th International Conference on Cloud  Computing, GRIDs, and Virtualization (Cloud Computing 2025), Valencia, Spain,  2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16392v1) [paper-pdf](http://arxiv.org/pdf/2503.16392v1)

**Authors**: Anket Mehra, Andreas Aßmuth, Malte Prieß

**Abstract**: With AI-based software becoming widely available, the risk of exploiting its capabilities, such as high automation and complex pattern recognition, could significantly increase. An AI used offensively to attack non-AI assets is referred to as offensive AI.   Current research explores how offensive AI can be utilized and how its usage can be classified. Additionally, methods for threat modeling are being developed for AI-based assets within organizations. However, there are gaps that need to be addressed. Firstly, there is a need to quantify the factors contributing to the AI threat. Secondly, there is a requirement to create threat models that analyze the risk of being attacked by AI for vulnerability assessment across all assets of an organization. This is particularly crucial and challenging in cloud environments, where sophisticated infrastructure and access control landscapes are prevalent. The ability to quantify and further analyze the threat posed by offensive AI enables analysts to rank vulnerabilities and prioritize the implementation of proactive countermeasures.   To address these gaps, this paper introduces the Graph of Effort, an intuitive, flexible, and effective threat modeling method for analyzing the effort required to use offensive AI for vulnerability exploitation by an adversary. While the threat model is functional and provides valuable support, its design choices need further empirical validation in future work.

摘要: 随着基于人工智能的软件变得广泛可用，利用其高自动化和复杂模式识别等能力的风险可能会显著增加。用来攻击非人工智能资产的人工智能被称为攻击性人工智能。目前的研究探索了如何利用攻击性人工智能，以及如何对其使用进行分类。此外，正在为组织内基于人工智能的资产开发威胁建模方法。然而，还有一些差距需要填补。首先，有必要量化造成人工智能威胁的因素。其次，需要创建威胁模型来分析被人工智能攻击的风险，以便对组织的所有资产进行漏洞评估。这在云环境中尤为关键和具有挑战性，因为在云环境中，复杂的基础设施和访问控制环境非常普遍。量化和进一步分析攻击性人工智能构成的威胁的能力使分析师能够对漏洞进行排名，并确定实施主动对策的优先顺序。为了解决这些差距，本文引入了努力图，这是一种直观、灵活和有效的威胁建模方法，用于分析使用攻击性人工智能进行对手漏洞利用所需的努力。虽然威胁模型是功能性的，并提供了有价值的支持，但其设计选择需要在未来的工作中进一步进行经验验证。



## **2. RESFL: An Uncertainty-Aware Framework for Responsible Federated Learning by Balancing Privacy, Fairness and Utility in Autonomous Vehicles**

RECFL：一个不确定性感知框架，通过平衡自动驾驶汽车中的隐私、公平和实用性，实现负责任的联邦学习 cs.LG

Submitted to PETS 2025 (under review)

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16251v1) [paper-pdf](http://arxiv.org/pdf/2503.16251v1)

**Authors**: Dawood Wasif, Terrence J. Moore, Jin-Hee Cho

**Abstract**: Autonomous vehicles (AVs) increasingly rely on Federated Learning (FL) to enhance perception models while preserving privacy. However, existing FL frameworks struggle to balance privacy, fairness, and robustness, leading to performance disparities across demographic groups. Privacy-preserving techniques like differential privacy mitigate data leakage risks but worsen fairness by restricting access to sensitive attributes needed for bias correction. This work explores the trade-off between privacy and fairness in FL-based object detection for AVs and introduces RESFL, an integrated solution optimizing both. RESFL incorporates adversarial privacy disentanglement and uncertainty-guided fairness-aware aggregation. The adversarial component uses a gradient reversal layer to remove sensitive attributes, reducing privacy risks while maintaining fairness. The uncertainty-aware aggregation employs an evidential neural network to weight client updates adaptively, prioritizing contributions with lower fairness disparities and higher confidence. This ensures robust and equitable FL model updates. We evaluate RESFL on the FACET dataset and CARLA simulator, assessing accuracy, fairness, privacy resilience, and robustness under varying conditions. RESFL improves detection accuracy, reduces fairness disparities, and lowers privacy attack success rates while demonstrating superior robustness to adversarial conditions compared to other approaches.

摘要: 自动驾驶汽车(AVs)越来越依赖联邦学习(FL)来增强感知模型，同时保护隐私。然而，现有的FL框架难以平衡隐私、公平和健壮性，导致不同人口群体之间的表现差异。像差异隐私这样的隐私保护技术降低了数据泄露风险，但限制了对偏见纠正所需的敏感属性的访问，从而恶化了公平性。该工作探讨了基于FL的AVS目标检测中的隐私和公平性之间的权衡，并介绍了一种优化两者的集成解决方案RESFL。RESFL融合了对抗性隐私解缠和不确定性引导的公平感知聚合。对抗性组件使用梯度反转层来移除敏感属性，从而在保持公平性的同时降低隐私风险。不确定性感知聚合使用证据神经网络自适应地对客户端更新进行加权，以较低的公平性差异和较高的置信度对贡献进行优先排序。这确保了稳健和公平的FL模型更新。我们在刻面数据集和CALA模拟器上对RESFL进行了评估，评估了在不同条件下的准确性、公平性、隐私弹性和健壮性。与其他方法相比，RESFL提高了检测精度，减少了公平性差异，降低了隐私攻击成功率，同时表现出对敌对条件的卓越稳健性。



## **3. AI Agents in Cryptoland: Practical Attacks and No Silver Bullet**

加密土地中的人工智能代理：实际攻击和没有银弹 cs.CR

12 pages, 8 figures

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16248v1) [paper-pdf](http://arxiv.org/pdf/2503.16248v1)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: The integration of AI agents with Web3 ecosystems harnesses their complementary potential for autonomy and openness, yet also introduces underexplored security risks, as these agents dynamically interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. Through empirical analysis of ElizaOS, a decentralized AI agent framework for automated Web3 operations, we demonstrate how adversaries can manipulate context by injecting malicious instructions into prompts or historical interaction records, leading to unintended asset transfers and protocol violations which could be financially devastating. Our findings indicate that prompt-based defenses are insufficient, as malicious inputs can corrupt an agent's stored context, creating cascading vulnerabilities across interactions and platforms. This research highlights the urgent need to develop AI agents that are both secure and fiduciarily responsible.

摘要: 人工智能代理与Web3生态系统的集成利用了它们在自治和开放方面的互补潜力，但也带来了未被探索的安全风险，因为这些代理与金融协议和一成不变的智能合同动态交互。本文研究了基于区块链的金融生态系统中人工智能代理在现实场景中面临对抗性威胁时的脆弱性。我们引入了上下文操纵的概念--一种利用未受保护的上下文面的综合攻击载体，包括输入通道、内存模块和外部数据馈送。通过对Elizabeth OS的经验分析，我们展示了攻击者如何通过向提示或历史交互记录中注入恶意指令来操纵上下文，从而导致意外的资产转移和协议违规，这可能是经济上的毁灭性破坏。我们的发现表明，基于提示的防御是不够的，因为恶意输入可以破坏代理的存储上下文，从而在交互和平台之间产生级联漏洞。这项研究突显了开发既安全又可信的人工智能代理的迫切需要。



## **4. Robust LLM safeguarding via refusal feature adversarial training**

通过拒绝功能对抗培训强大的LLM保障 cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.20089v2) [paper-pdf](http://arxiv.org/pdf/2409.20089v2)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.

摘要: 大型语言模型(LLM)很容易受到可能引起有害响应的对抗性攻击。由于越狱机制的不透明性和强大训练LLM的高计算成本，防御此类攻击仍然具有挑战性。我们证明了对抗性攻击共享一个通用的机制来规避LLM安全机制，该机制通过在剩余流嵌入空间中消融一个称为拒绝特征的维度来工作。我们进一步证明了拒绝特征消融(RFA)的操作近似于补偿模型安全性的最坏情况的扰动。基于这些发现，我们提出了拒绝特征对抗训练(Refat)，这是一种通过RFA模拟输入级攻击的效果来高效执行LLM对抗训练的新算法。实验结果表明，与现有的对抗性训练方法相比，REFAT显著地提高了三种流行的LLMS对多种对抗性攻击的健壮性，并且具有相当少的计算开销。



## **5. 2DSig-Detect: a semi-supervised framework for anomaly detection on image data using 2D-signatures**

2DSig-Detect：使用2D签名对图像数据进行异常检测的半监督框架 cs.CV

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.04982v2) [paper-pdf](http://arxiv.org/pdf/2409.04982v2)

**Authors**: Xinheng Xie, Kureha Yamaguchi, Margaux Leblanc, Simon Malzard, Varun Chhabra, Victoria Nockles, Yue Wu

**Abstract**: The rapid advancement of machine learning technologies raises questions about the security of machine learning models, with respect to both training-time (poisoning) and test-time (evasion, impersonation, and inversion) attacks. Models performing image-related tasks, e.g. detection, and classification, are vulnerable to adversarial attacks that can degrade their performance and produce undesirable outcomes. This paper introduces a novel technique for anomaly detection in images called 2DSig-Detect, which uses a 2D-signature-embedded semi-supervised framework rooted in rough path theory. We demonstrate our method in adversarial settings for training-time and test-time attacks, and benchmark our framework against other state of the art methods. Using 2DSig-Detect for anomaly detection, we show both superior performance and a reduction in the computation time to detect the presence of adversarial perturbations in images.

摘要: 机器学习技术的快速发展引发了有关机器学习模型安全性的问题，包括训练时（中毒）和测试时（逃避、模仿和倒置）攻击。执行图像相关任务（例如检测和分类）的模型很容易受到对抗攻击，这些攻击可能会降低其性能并产生不良结果。本文介绍了一种名为2DSig-Detect的图像异常检测新技术，该技术使用植根于粗糙路径理论的2D签名嵌入半监督框架。我们在训练时和测试时攻击的对抗环境中展示了我们的方法，并将我们的框架与其他最先进的方法进行基准测试。使用2DSig-Detect进行异常检测，我们既表现出卓越的性能，又减少了检测图像中存在对抗性扰动的计算时间。



## **6. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders**

SAUCE：使用稀疏自动编码器的视觉语言模型中的选择性概念消除 cs.CV

More comparative experiments are needed

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.14530v2) [paper-pdf](http://arxiv.org/pdf/2503.14530v2)

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.

摘要: 视觉语言模型(VLM)的遗忘方法主要采用来自大型语言模型(LLM)的技术，依赖于需要大量注释遗忘集的权重更新。此外，这些方法在粗粒度上执行遗忘，经常导致过度遗忘和降低模型效用。为了解决这个问题，我们引入了SASE，这是一种新的方法，它利用稀疏自动编码器(SAE)在VLM中进行细粒度和选择性的概念遗忘。简而言之，SASE首先训练SAE捕获高维的、语义丰富的稀疏特征。然后确定与目标概念最相关的特征以进行遗忘。在推理过程中，它有选择地修改这些特征以抑制特定概念，同时保留不相关的信息。我们在两个不同的VLM，LLaVA-v1.5-7B和Llama-3.2-11B-Vision-Indict上评估SAUE，跨越两种类型的任务：具体概念遗忘(物体和运动场景)和抽象概念遗忘(情绪、颜色和材料)，总共包含60个概念。大量的实验表明，在保持可比的模型效用的情况下，SASE在遗忘质量方面比最先进的方法高出18.04%。此外，我们还研究了SASE对广泛使用的敌意攻击的健壮性、其跨模型的可转移性以及其在处理多个并发遗忘请求时的可扩展性。我们的研究结果表明，SASE是一种有效且可扩展的解决方案，可用于解决VLMS中的选择性概念遗忘问题。



## **7. DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence**

DroidTTP：使用TTP映射Android应用程序以实现网络威胁情报 cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15866v1) [paper-pdf](http://arxiv.org/pdf/2503.15866v1)

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Abstract**: The widespread adoption of Android devices for sensitive operations like banking and communication has made them prime targets for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods rely on binary classification, failing to provide insights into adversarial Tactics, Techniques, and Procedures (TTPs). Understanding malware behavior is crucial for enhancing cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework. Our curated dataset explicitly links MITRE TTPs to Android applications. We developed an automated solution leveraging the Problem Transformation Approach (PTA) and Large Language Models (LLMs) to map applications to both Tactics and Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with prompt engineering and LLM fine-tuning for TTP predictions. Our structured pipeline includes dataset creation, hyperparameter tuning, data augmentation, feature selection, model development, and SHAP-based model interpretability. Among LLMs, Llama achieved the highest performance in Tactic classification with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs, achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753 for Technique classification, with a Hamming Loss of 0.0054 and 0.0050, respectively. While XGBoost showed superior performance, the narrow margin highlights the potential of LLM-based approaches in TTP classification.

摘要: Android设备广泛用于银行和通信等敏感操作，使其成为网络威胁的主要目标，特别是高级持久性威胁(APT)和复杂的恶意软件攻击。传统的恶意软件检测方法依赖于二进制分类，无法提供对敌对战术、技术和过程(TTP)的洞察。了解恶意软件行为对于加强网络安全防御至关重要。为了弥补这一差距，我们在MITRE ATT&CK框架的基础上引入了DroidTTP，一个将Android恶意软件行为映射到TTP的框架。我们精心挑选的数据集明确地将MITRE TTP链接到Android应用程序。我们开发了一个自动化解决方案，利用问题转换方法(PTA)和大型语言模型(LLM)将应用程序映射到战术和技术。此外，我们使用了具有即时工程和LLM微调的检索-增强生成(RAG)来进行TTP预测。我们的结构化流程包括数据集创建、超参数调整、数据增强、特征选择、模型开发和基于Shap的模型可解释性。在LLMS中，大羊驼在战术分类上表现最好，贾卡德相似度为0.9583，Hamming损失为0.0182；在技术分类上表现最好，Jaccard相似度为0.9348，Hamming损失为0.0127。然而，标签Powerset XGBoost模型的表现优于LLMS，战术分类的Jaccard相似度为0.9893，技术分类的Jaccard相似度为0.9753，Hamming损失分别为0.0054和0.0050。虽然XGBoost表现出了优越的性能，但狭窄的差距突显了基于LLM的方法在TTP分类中的潜力。



## **8. Cyber Threats in Financial Transactions -- Addressing the Dual Challenge of AI and Quantum Computing**

金融交易中的网络威胁--应对人工智能和量子计算的双重挑战 cs.CR

38 Pages, 3 tables, Technical Report,  https://www.acfti.org/cftirc-community/technical-report-1-quantum-finance-cyber-threats

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.15678v1) [paper-pdf](http://arxiv.org/pdf/2503.15678v1)

**Authors**: Ahmed M. Elmisery, Mirela Sertovic, Andrew Zayin, Paul Watson

**Abstract**: The financial sector faces escalating cyber threats amplified by artificial intelligence (AI) and the advent of quantum computing. AI is being weaponized for sophisticated attacks like deepfakes and AI-driven malware, while quantum computing threatens to render current encryption methods obsolete. This report analyzes these threats, relevant frameworks, and possible countermeasures like quantum cryptography. AI enhances social engineering and phishing attacks via personalized content, lowers entry barriers for cybercriminals, and introduces risks like data poisoning and adversarial AI. Quantum computing, particularly Shor's algorithm, poses a fundamental threat to current encryption standards (RSA and ECC), with estimates suggesting cryptographically relevant quantum computers could emerge within the next 5-30 years. The "harvest now, decrypt later" scenario highlights the urgency of transitioning to quantum-resistant cryptography. This is key. Existing legal frameworks are evolving to address AI in cybercrime, but quantum threats require new initiatives. International cooperation and harmonized regulations are crucial. Quantum Key Distribution (QKD) offers theoretical security but faces practical limitations. Post-quantum cryptography (PQC) is a promising alternative, with ongoing standardization efforts. Recommendations for international regulators include fostering collaboration and information sharing, establishing global standards, supporting research and development in quantum security, harmonizing legal frameworks, promoting cryptographic agility, and raising awareness and education. The financial industry must adopt a proactive and adaptive approach to cybersecurity, investing in research, developing migration plans for quantum-resistant cryptography, and embracing a multi-faceted, collaborative strategy to build a resilient, quantum-safe, and AI-resilient financial ecosystem

摘要: 金融行业面临不断升级的网络威胁，人工智能(AI)和量子计算的出现放大了这一威胁。人工智能正被武器化，用于深度假冒和人工智能驱动的恶意软件等复杂攻击，而量子计算可能会使当前的加密方法过时。这份报告分析了这些威胁、相关框架和可能的对策，如量子密码学。人工智能通过个性化内容加强社会工程和网络钓鱼攻击，降低网络犯罪分子的进入门槛，并引入数据中毒和对抗性人工智能等风险。量子计算，特别是肖尔的算法，对当前的加密标准(RSA和ECC)构成了根本威胁，估计表明，在未来5-30年内，可能会出现与密码相关的量子计算机。“现在收获，以后解密”的情景凸显了向量子抵抗密码术过渡的紧迫性。这是关键。现有的法律框架正在演变，以解决网络犯罪中的人工智能问题，但量子威胁需要新的举措。国际合作和统一的条例至关重要。量子密钥分发(QKD)提供了理论上的安全性，但也面临着实践上的局限性。随着标准化工作的进行，后量子密码学(PQC)是一种很有前途的替代方案。对国际监管机构的建议包括促进合作和信息共享，建立全球标准，支持量子安全方面的研究和开发，协调法律框架，促进加密灵活性，以及提高认识和教育。金融行业必须对网络安全采取主动和自适应的方法，投资于研究，制定量子耐加密的迁移计划，并采用多方面的协作战略，以构建具有弹性、量子安全和人工智能弹性的金融生态系统



## **9. No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data**

不，我当然可以！使用无害微调数据可以利用拒绝机制 cs.CR

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.19537v2) [paper-pdf](http://arxiv.org/pdf/2502.19537v2)

**Authors**: Joshua Kazdan, Lisa Yu, Rylan Schaeffer, Chris Cundy, Sanmi Koyejo, Krishnamurthy Dvijotham

**Abstract**: Leading language model (LM) providers like OpenAI and Google offer fine-tuning APIs that allow customers to adapt LMs for specific use cases. To prevent misuse, these LM providers implement filtering mechanisms to block harmful fine-tuning data. Consequently, adversaries seeking to produce unsafe LMs via these APIs must craft adversarial training data that are not identifiably harmful. We make three contributions in this context: 1. We show that many existing attacks that use harmless data to create unsafe LMs rely on eliminating model refusals in the first few tokens of their responses. 2. We show that such prior attacks can be blocked by a simple defense that pre-fills the first few tokens from an aligned model before letting the fine-tuned model fill in the rest. 3. We describe a new data-poisoning attack, ``No, Of course I Can Execute'' (NOICE), which exploits an LM's formulaic refusal mechanism to elicit harmful responses. By training an LM to refuse benign requests on the basis of safety before fulfilling those requests regardless, we are able to jailbreak several open-source models and a closed-source model (GPT-4o). We show an attack success rate (ASR) of 57% against GPT-4o; our attack earned a Bug Bounty from OpenAI. Against open-source models protected by simple defenses, we improve ASRs by an average of 3.25 times compared to the best performing previous attacks that use only harmless data. NOICE demonstrates the exploitability of repetitive refusal mechanisms and broadens understanding of the threats closed-source models face from harmless data.

摘要: 领先的语言模型(LM)提供商，如OpenAI和Google，提供了微调的API，允许客户根据特定的用例调整LMS。为防止误用，这些LM提供程序实施过滤机制以阻止有害的微调数据。因此，试图通过这些API生成不安全的LMS的攻击者必须创建不能识别有害的对抗性训练数据。我们在这方面做了三点贡献：1.我们证明了许多现有的使用无害数据来创建不安全的LMS的攻击依赖于消除其响应的前几个令牌中的模型拒绝。2.我们证明了这样的先前攻击可以通过一个简单的防御来阻止，即预先填充来自对齐模型的前几个令牌，然后让微调的模型填充其余的令牌。3.我们描述了一种新的数据中毒攻击，``不，我当然可以执行‘’(Noice)，它利用LM的公式化拒绝机制来引发有害的响应。通过训练LM在满足良性请求之前出于安全考虑拒绝这些请求，我们能够越狱几个开源模型和一个封闭源代码模型(GPT-40)。我们对GPT-40的攻击成功率(ASR)为57%；我们的攻击从OpenAI获得了错误赏金。相对于由简单防御保护的开源模型，我们将ASR平均提高了3.25倍，而之前仅使用无害数据的攻击性能最好。Noice展示了重复拒绝机制的可利用性，并拓宽了对封闭源代码模型面临的无害数据威胁的理解。



## **10. Safety at Scale: A Comprehensive Survey of Large Model Safety**

大规模安全性：大型车型安全性全面调查 cs.CR

47 pages, 3 figures, 11 tables; GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.05206v3) [paper-pdf](http://arxiv.org/pdf/2502.05206v3)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.

摘要: 大型模型的快速发展，受到其通过大规模预训练而具有的非凡学习和泛化能力的推动，重塑了人工智能(AI)的版图。这些模型现在是广泛应用的基础，包括对话式人工智能、推荐系统、自动驾驶、内容生成、医疗诊断和科学发现。然而，它们的广泛部署也使它们面临重大的安全风险，引发了人们对健壮性、可靠性和道德影响的担忧。本调查系统地回顾了当前关于大模型的安全研究，包括视觉基础模型(VFM)、大语言模型(LLMS)、视觉语言预训练(VLP)模型、视觉语言模型(VLMS)、扩散模型(DM)和基于大模型的代理。我们的工作总结如下：(1)对这些模型的安全威胁进行了全面的分类，包括对抗性攻击、数据中毒、后门攻击、越狱和快速注入攻击、能量延迟攻击、数据和模型提取攻击以及新出现的特定于代理的威胁。(2)我们回顾了针对每种攻击类型提出的防御策略(如果可用)，并总结了安全研究常用的数据集和基准。(3)在此基础上，我们确定并讨论了大型模型安全方面的开放挑战，强调需要全面的安全评估、可扩展和有效的防御机制以及可持续的数据实践。更重要的是，我们强调了研究界和国际合作集体努力的必要性。我们的工作可以作为研究人员和从业者的有用参考，促进正在进行的全面防御系统和平台的开发，以保护人工智能模型。



## **11. Adaptive Pruning with Module Robustness Sensitivity: Balancing Compression and Robustness**

具有模块稳健性敏感性的自适应修剪：平衡压缩和稳健性 cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2410.15176v2) [paper-pdf](http://arxiv.org/pdf/2410.15176v2)

**Authors**: Lincen Bai, Hedi Tabia, Raúl Santos-Rodríguez

**Abstract**: Neural network pruning has traditionally focused on weight-based criteria to achieve model compression, frequently overlooking the crucial balance between adversarial robustness and accuracy. Existing approaches often fail to preserve robustness in pruned networks, leaving them more susceptible to adversarial attacks. This paper introduces Module Robustness Sensitivity (MRS), a novel metric that quantifies layer-wise sensitivity to adversarial perturbations and dynamically informs pruning decisions. Leveraging MRS, we propose Module Robust Pruning and Fine-Tuning (MRPF), an adaptive pruning algorithm compatible with any adversarial training method, offering both flexibility and scalability. Extensive experiments on SVHN, CIFAR, and Tiny-ImageNet across diverse architectures, including ResNet, VGG, and MobileViT, demonstrate that MRPF significantly enhances adversarial robustness while maintaining competitive accuracy and computational efficiency. Furthermore, MRPF consistently outperforms state-of-the-art structured pruning methods in balancing robustness, accuracy, and compression. This work establishes a practical and generalizable framework for robust pruning, addressing the long-standing trade-off between model compression and robustness preservation.

摘要: 传统上，神经网络剪枝侧重于基于权重的标准来实现模型压缩，但往往忽略了对手健壮性和准确性之间的关键平衡。现有的方法往往不能在经过剪枝的网络中保持健壮性，从而使它们更容易受到对手攻击。本文引入了模块健壮性敏感度(MRS)，这是一种新的度量，它量化了对对手扰动的层级敏感度，并动态地通知剪枝决策。利用MRS，我们提出了模块稳健剪枝和精调(MRPF)，这是一种与任何对抗性训练方法兼容的自适应剪枝算法，提供了灵活性和可扩展性。在包括ResNet、VGG和MobileViT在内的不同体系结构上对SVHN、CIFAR和Tiny-ImageNet进行的大量实验表明，MRPF在保持竞争精度和计算效率的同时，显著增强了对手的健壮性。此外，在稳健性、准确性和压缩方面，MRPF始终优于最先进的结构化剪枝方法。这项工作建立了一个实用和可推广的稳健剪枝框架，解决了模型压缩和稳健性保持之间的长期权衡。



## **12. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

通过动态最大化优化改进普遍对抗扰动的推广 cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.12793v2) [paper-pdf](http://arxiv.org/pdf/2503.12793v2)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.

摘要: 深度神经网络(DNN)容易受到普遍的对抗性扰动(UAP)的影响。这些扰动是精心设计的，目的是在所有样本类中普遍欺骗目标模型。与实例特定的对抗性示例(AE)不同，生成UAP更加复杂，因为它们必须在广泛的数据样本和模型中推广。我们的研究表明，现有的通用攻击方法使用带有静态模型参数快照的DNN来优化UAP，没有充分利用DNN的潜力来生成更有效的UAP。我们建议使用动态模型-数据对来生成UAP，而不是针对具有固定训练集的静态DNN模型来优化UAP。特别是，我们引入了动态最大优化策略，旨在对UAP进行各种最优模型-数据对的优化。我们称这种方法为DM-UAP。DM-UAP使用迭代的最大-最小-最小优化框架来细化模型-数据对，并结合课程UAP学习算法来彻底检查模型参数和数据的组合空间。在ImageNet数据集上的综合实验表明，DM-UAP显著增强了UAP的跨样本普适性和跨模型可转移性。仅使用500个样本生成UAP，DM-UAP的性能优于最先进的方法，平均傻瓜率提高了12.108%。



## **13. Robustness bounds on the successful adversarial examples in probabilistic models: Implications from Gaussian processes**

概率模型中成功对抗示例的鲁棒性界限：高斯过程的含义 cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2403.01896v2) [paper-pdf](http://arxiv.org/pdf/2403.01896v2)

**Authors**: Hiroaki Maeshima, Akira Otsuka

**Abstract**: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification, a probabilistic inference model. We proved a new upper bound of the probability of a successful AE attack that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.

摘要: 对抗示例（AE）是机器学习的一种攻击方法，通过向数据添加难以察觉的扰动来制作，从而导致错误分类。在本文中，我们基于高斯过程（GP）分类（一种概率推理模型）研究了成功AE的概率的上限。我们证明了AE攻击成功概率的新上界，该上界取决于AE的扰动规范、GP中使用的核函数以及训练数据集中具有不同标签的最接近对的距离。令人惊讶的是，无论样本数据集的分布如何，上限都会确定。我们表明我们的理论结果通过使用ImageNet的实验得到了证实。此外，我们还表明，改变核函数的参数会导致AE成功概率上界的变化。



## **14. A Semantic and Clean-label Backdoor Attack against Graph Convolutional Networks**

针对图卷积网络的语义和干净标签后门攻击 cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.14922v1) [paper-pdf](http://arxiv.org/pdf/2503.14922v1)

**Authors**: Jiazhu Dai, Haoyu Sun

**Abstract**: Graph Convolutional Networks (GCNs) have shown excellent performance in graph-structured tasks such as node classification and graph classification. However, recent research has shown that GCNs are vulnerable to a new type of threat called the backdoor attack, where the adversary can inject a hidden backdoor into the GCNs so that the backdoored model performs well on benign samples, whereas its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. Clean-label backdoor attack and semantic backdoor attack are two new backdoor attacks to Deep Neural Networks (DNNs), they are more imperceptible and have posed new and serious threats. The semantic and clean-label backdoor attack is not fully explored in GCNs. In this paper, we propose a semantic and clean-label backdoor attack against GCNs under the context of graph classification to reveal the existence of this security vulnerability in GCNs. Specifically, SCLBA conducts an importance analysis on graph samples to select one type of node as semantic trigger, which is then inserted into the graph samples to create poisoning samples without changing the labels of the poisoning samples to the attacker-specified target label. We evaluate SCLBA on multiple datasets and the results show that SCLBA can achieve attack success rates close to 99% with poisoning rates of less than 3%, and with almost no impact on the performance of model on benign samples.

摘要: 图卷积网络(GCNS)在节点分类、图分类等图结构任务中表现出优异的性能。然而，最近的研究表明，GCNS容易受到一种称为后门攻击的新型威胁的攻击，在这种威胁中，攻击者可以向GCNS注入隐藏的后门，以便后门模型在良性样本上执行得很好，而如果隐藏的后门被攻击者定义的触发器激活，则其预测将被恶意更改为攻击者指定的目标标签。干净标签后门攻击和语义后门攻击是深度神经网络的两种新的后门攻击，它们的隐蔽性更强，已经构成了新的严重威胁。语义和干净标签的后门攻击在GCNS中没有得到充分的探索。为了揭示GCNS中存在的安全漏洞，提出了一种基于图分类的语义和干净标签的GCNS后门攻击方法。具体地说，SCLBA对图样本进行重要性分析，选择一种类型的节点作为语义触发器，然后将其插入到图样本中创建中毒样本，而不会将中毒样本的标签更改为攻击者指定的目标标签。我们在多个数据集上对SCLBA进行了评估，结果表明，SCLBA可以达到接近99%的攻击成功率，而投毒率低于3%，并且对良性样本的模型性能几乎没有影响。



## **15. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

ADBM：用于可靠对抗净化的对抗扩散桥模型 cs.LG

ICLR 2025, fix typos in the proof

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2408.00315v4) [paper-pdf](http://arxiv.org/pdf/2408.00315v4)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.

摘要: 最近，基于扩散的纯化（DiffPure）被认为是针对对抗性例子的有效防御方法。然而，我们发现直接使用原始预训练的扩散模型进行对抗性纯化的迪夫Pure是次优的。这是由于噪音净化性能和数据恢复质量之间固有的权衡。此外，现有的DistPure评估的可靠性值得怀疑，因为它们依赖于弱适应性攻击。在这项工作中，我们提出了一种新型的对抗扩散桥模型，称为ADBM。ADBM直接构建了从扩散的对抗数据到其原始干净示例的反向桥梁，增强了原始扩散模型的净化能力。通过各种场景的理论分析和实验验证，ADBM已被证明是一种卓越且强大的防御机制，为实际应用提供了巨大的前景。



## **16. Synthesizing Grid Data with Cyber Resilience and Privacy Guarantees**

综合具有网络弹性和隐私保证的网格数据 eess.SY

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.14877v1) [paper-pdf](http://arxiv.org/pdf/2503.14877v1)

**Authors**: Shengyang Wu, Vladimir Dvorkin

**Abstract**: Differential privacy (DP) provides a principled approach to synthesizing data (e.g., loads) from real-world power systems while limiting the exposure of sensitive information. However, adversaries may exploit synthetic data to calibrate cyberattacks on the source grids. To control these risks, we propose new DP algorithms for synthesizing data that provide the source grids with both cyber resilience and privacy guarantees. The algorithms incorporate both normal operation and attack optimization models to balance the fidelity of synthesized data and cyber resilience. The resulting post-processing optimization is reformulated as a robust optimization problem, which is compatible with the exponential mechanism of DP to moderate its computational burden.

摘要: 差异隐私（DP）提供了一种有原则的方法来合成数据（例如，负载）来自现实世界的电力系统，同时限制敏感信息的暴露。然而，对手可能会利用合成数据来校准对源网格的网络攻击。为了控制这些风险，我们提出了新的DP算法来合成数据，为源网格提供网络弹性和隐私保证。这些算法结合了正常操作和攻击优化模型，以平衡合成数据的保真度和网络弹性。所得的后处理优化被重新表述为鲁棒优化问题，该问题与DP的指数机制兼容，以减轻其计算负担。



## **17. Temporal Context Awareness: A Defense Framework Against Multi-turn Manipulation Attacks on Large Language Models**

时间上下文感知：针对大型语言模型多轮操纵攻击的防御框架 cs.CR

6 pages, 2 figures, IEEE CAI

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15560v1) [paper-pdf](http://arxiv.org/pdf/2503.15560v1)

**Authors**: Prashant Kulkarni, Assaf Namer

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to sophisticated multi-turn manipulation attacks, where adversaries strategically build context through seemingly benign conversational turns to circumvent safety measures and elicit harmful or unauthorized responses. These attacks exploit the temporal nature of dialogue to evade single-turn detection methods, representing a critical security vulnerability with significant implications for real-world deployments.   This paper introduces the Temporal Context Awareness (TCA) framework, a novel defense mechanism designed to address this challenge by continuously analyzing semantic drift, cross-turn intention consistency and evolving conversational patterns. The TCA framework integrates dynamic context embedding analysis, cross-turn consistency verification, and progressive risk scoring to detect and mitigate manipulation attempts effectively. Preliminary evaluations on simulated adversarial scenarios demonstrate the framework's potential to identify subtle manipulation patterns often missed by traditional detection techniques, offering a much-needed layer of security for conversational AI systems. In addition to outlining the design of TCA , we analyze diverse attack vectors and their progression across multi-turn conversation, providing valuable insights into adversarial tactics and their impact on LLM vulnerabilities. Our findings underscore the pressing need for robust, context-aware defenses in conversational AI systems and highlight TCA framework as a promising direction for securing LLMs while preserving their utility in legitimate applications. We make our implementation available to support further research in this emerging area of AI security.

摘要: 大型语言模型(LLM)越来越容易受到复杂的多回合操纵攻击，在这种攻击中，对手通过看似良性的对话转向来策略性地构建上下文，以绕过安全措施并引发有害或未经授权的响应。这些攻击利用对话的时间性来逃避单轮检测方法，这是一个严重的安全漏洞，对现实世界的部署具有重大影响。本文介绍了时态语境感知(TCA)框架，这是一种新的防御机制，旨在通过不断分析语义漂移、跨话轮意图一致性和会话模式演变来应对这一挑战。TCA框架集成了动态上下文嵌入分析、跨回合一致性验证和渐进式风险评分，以有效检测和减少操纵企图。对模拟对抗性场景的初步评估表明，该框架有可能识别传统检测技术经常遗漏的细微操纵模式，为对话式人工智能系统提供了亟需的安全层。除了概述TCA的设计之外，我们还分析了不同的攻击向量及其在多轮对话中的进展，为敌方战术及其对LLM漏洞的影响提供了有价值的见解。我们的发现强调了在对话式人工智能系统中对强大的、上下文感知的防御的迫切需要，并强调TCA框架是一个有希望的方向，可以在保护LLM的有效性的同时保护其在合法应用中的有效性。我们提供我们的实现，以支持在这一新兴的人工智能安全领域的进一步研究。



## **18. Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection**

多轮对话中社会工程的个性化攻击-- LLM模拟和检测代理 cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15552v1) [paper-pdf](http://arxiv.org/pdf/2503.15552v1)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大语言模型(LLM)驱动的聊天机器人，构成了社交媒体平台上的社交工程(SE)攻击的巨大风险。基于聊天的多轮交互中的SE检测比单实例检测要复杂得多，这是因为这些对话的动态性质。缓解这一威胁的一个关键因素是了解SE攻击的运作机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何导致他们的易感性。在这项工作中，我们提出了一个LLM代理框架SE-VSim，通过生成多话轮会话来模拟SE攻击机制。我们对具有不同个性特征的受害者代理进行建模，以评估心理特征如何影响操纵的易感性。使用1000多个模拟对话的数据集，我们检查了攻击场景，在这些场景中，伪装成招聘者、资助机构和记者的对手试图提取敏感信息。基于这一分析，我们提出了一个概念证明，SE-OmniGuard，通过利用受害者个性的先验知识，评估攻击策略，并监控对话中的信息交换来识别潜在的SE尝试，为用户提供个性化保护。



## **19. Adversarial Robustness in Parameter-Space Classifiers**

参数空间分类器中的对抗鲁棒性 cs.LG

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2502.20314v2) [paper-pdf](http://arxiv.org/pdf/2502.20314v2)

**Authors**: Tamir Shor, Ethan Fetaya, Chaim Baskin, Alex Bronstein

**Abstract**: Implicit Neural Representations (INRs) have been recently garnering increasing interest in various research fields, mainly due to their ability to represent large, complex data in a compact and continuous manner. Past work further showed that numerous popular downstream tasks can be performed directly in the INR parameter-space. Doing so can substantially reduce the computational resources required to process the represented data in their native domain. A major difficulty in using modern machine-learning approaches, is their high susceptibility to adversarial attacks, which have been shown to greatly limit the reliability and applicability of such methods in a wide range of settings. In this work, we show that parameter-space models trained for classification are inherently robust to adversarial attacks -- without the need of any robust training. To support our claims, we develop a novel suite of adversarial attacks targeting parameter-space classifiers, and furthermore analyze practical considerations of attacking parameter-space classifiers.

摘要: 隐式神经表示(INR)最近在各个研究领域引起了越来越多的兴趣，这主要是因为它们能够以紧凑和连续的方式表示大型、复杂的数据。过去的工作进一步表明，许多流行的下游任务可以直接在INR参数空间中执行。这样做可以大大减少在其本地域中处理所表示的数据所需的计算资源。使用现代机器学习方法的一个主要困难是它们对对手攻击的高度敏感性，这已被证明在广泛的环境中极大地限制了这种方法的可靠性和适用性。在这项工作中，我们证明了为分类而训练的参数空间模型在本质上对对手攻击是健壮的--而不需要任何健壮的训练。为了支持我们的观点，我们开发了一套新的针对参数空间分类器的对抗性攻击，并进一步分析了攻击参数空间分类器的实际考虑。



## **20. Anomaly-Flow: A Multi-domain Federated Generative Adversarial Network for Distributed Denial-of-Service Detection**

异常流：一种用于分布式拒绝服务检测的多域联邦生成对抗网络 cs.CR

8 pages, 4 figures

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14618v1) [paper-pdf](http://arxiv.org/pdf/2503.14618v1)

**Authors**: Leonardo Henrique de Melo, Gustavo de Carvalho Bertoli, Michele Nogueira, Aldri Luiz dos Santos, Lourenço Alves Pereira Junior

**Abstract**: Distributed denial-of-service (DDoS) attacks remain a critical threat to Internet services, causing costly disruptions. While machine learning (ML) has shown promise in DDoS detection, current solutions struggle with multi-domain environments where attacks must be detected across heterogeneous networks and organizational boundaries. This limitation severely impacts the practical deployment of ML-based defenses in real-world settings.   This paper introduces Anomaly-Flow, a novel framework that addresses this critical gap by combining Federated Learning (FL) with Generative Adversarial Networks (GANs) for privacy-preserving, multi-domain DDoS detection. Our proposal enables collaborative learning across diverse network domains while preserving data privacy through synthetic flow generation. Through extensive evaluation across three distinct network datasets, Anomaly-Flow achieves an average F1-score of $0.747$, outperforming baseline models. Importantly, our framework enables organizations to share attack detection capabilities without exposing sensitive network data, making it particularly valuable for critical infrastructure and privacy-sensitive sectors.   Beyond immediate technical contributions, this work provides insights into the challenges and opportunities in multi-domain DDoS detection, establishing a foundation for future research in collaborative network defense systems. Our findings have important implications for academic research and industry practitioners working to deploy practical ML-based security solutions.

摘要: 分布式拒绝服务(DDoS)攻击仍然是对互联网服务的严重威胁，造成代价高昂的中断。虽然机器学习(ML)在DDoS检测方面显示出了希望，但当前的解决方案在多域环境中苦苦挣扎，在多域环境中，必须跨不同的网络和组织边界检测攻击。这一限制严重影响了基于ML的防御在现实世界环境中的实际部署。本文介绍了一种新的框架，它通过将联邦学习(FL)和生成性对抗网络(GANS)相结合来解决这一关键缺陷，以实现隐私保护的多域DDoS检测。我们的建议支持跨不同网络领域的协作学习，同时通过合成流量生成保护数据隐私。通过对三个不同的网络数据集进行广泛的评估，异常流实现了0.747美元的F1平均得分，表现优于基线模型。重要的是，我们的框架使组织能够在不暴露敏感网络数据的情况下共享攻击检测功能，使其对关键基础设施和隐私敏感部门特别有价值。除了直接的技术贡献外，这项工作还提供了对多域DDoS检测的挑战和机遇的见解，为未来协作网络防御系统的研究奠定了基础。我们的发现对致力于部署实用的基于ML的安全解决方案的学术研究和行业从业者具有重要的意义。



## **21. VGFL-SA: Vertical Graph Federated Learning Structure Attack Based on Contrastive Learning**

VGFL-SA：基于对比学习的垂直图联邦学习结构攻击 cs.LG

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2502.16793v2) [paper-pdf](http://arxiv.org/pdf/2502.16793v2)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks (GNNs) have gained attention for their ability to learn representations from graph data. Due to privacy concerns and conflicts of interest that prevent clients from directly sharing graph data with one another, Vertical Graph Federated Learning (VGFL) frameworks have been developed. Recent studies have shown that VGFL is vulnerable to adversarial attacks that degrade performance. However, it is a common problem that client nodes are often unlabeled in the realm of VGFL. Consequently, the existing attacks, which rely on the availability of labeling information to obtain gradients, are inherently constrained in their applicability. This limitation precludes their deployment in practical, real-world environments. To address the above problems, we propose a novel graph adversarial attack against VGFL, referred to as VGFL-SA, to degrade the performance of VGFL by modifying the local clients structure without using labels. Specifically, VGFL-SA uses a contrastive learning method to complete the attack before the local clients are trained. VGFL-SA first accesses the graph structure and node feature information of the poisoned clients, and generates the contrastive views by node-degree-based edge augmentation and feature shuffling augmentation. Then, VGFL-SA uses the shared graph encoder to get the embedding of each view, and the gradients of the adjacency matrices are obtained by the contrastive function. Finally, perturbed edges are generated using gradient modification rules. We validated the performance of VGFL-SA by performing a node classification task on real-world datasets, and the results show that VGFL-SA achieves good attack effectiveness and transferability.

摘要: 图形神经网络(GNN)因其从图形数据中学习表示的能力而受到关注。由于隐私问题和利益冲突阻碍了客户之间直接共享图形数据，垂直图形联合学习(VGFL)框架已经开发出来。最近的研究表明，VGFL很容易受到降低性能的对抗性攻击。然而，在VGFL领域中，一个常见的问题是客户端节点通常是未标记的。因此，现有的攻击依赖于标记信息的可用性来获得梯度，其适用性受到固有的限制。这一限制排除了它们在实际、真实环境中的部署。针对上述问题，我们提出了一种新的针对VGFL的图对抗攻击，称为VGFL-SA，通过修改本地客户端结构而不使用标签来降低VGFL的性能。具体地说，VGFL-SA使用对比学习方法在本地客户端训练之前完成攻击。VGFL-SA首先获取中毒客户端的图结构和节点特征信息，然后通过基于节点度的边增强和特征置乱增强生成对比视图。然后，VGFL-SA使用共享图编码器得到每个视点的嵌入，并通过对比函数得到邻接矩阵的梯度。最后，使用梯度修正规则生成扰动边缘。我们通过在真实数据集上执行节点分类任务来验证VGFL-SA的性能，结果表明VGFL-SA具有良好的攻击有效性和可转移性。



## **22. Unveiling the Role of Randomization in Multiclass Adversarial Classification: Insights from Graph Theory**

揭示随机化在多类对抗分类中的作用：来自图论的见解 cs.LG

9 pages (main), 30 in total. Camera-ready version, accepted at  AISTATS 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14299v1) [paper-pdf](http://arxiv.org/pdf/2503.14299v1)

**Authors**: Lucas Gnecco-Heredia, Matteo Sammut, Muni Sreenivas Pydi, Rafael Pinot, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Randomization as a mean to improve the adversarial robustness of machine learning models has recently attracted significant attention. Unfortunately, much of the theoretical analysis so far has focused on binary classification, providing only limited insights into the more complex multiclass setting. In this paper, we take a step toward closing this gap by drawing inspiration from the field of graph theory. Our analysis focuses on discrete data distributions, allowing us to cast the adversarial risk minimization problems within the well-established framework of set packing problems. By doing so, we are able to identify three structural conditions on the support of the data distribution that are necessary for randomization to improve robustness. Furthermore, we are able to construct several data distributions where (contrarily to binary classification) switching from a deterministic to a randomized solution significantly reduces the optimal adversarial risk. These findings highlight the crucial role randomization can play in enhancing robustness to adversarial attacks in multiclass classification.

摘要: 随机化作为提高机器学习模型对抗性稳健性的一种手段，最近引起了人们的广泛关注。不幸的是，到目前为止，许多理论分析都集中在二进制分类上，对更复杂的多类设置只提供了有限的见解。在本文中，我们从图论领域汲取灵感，朝着缩小这一差距迈出了一步。我们的分析集中在离散数据分布上，允许我们在集合打包问题的良好框架内求解对抗性风险最小化问题。通过这样做，我们能够确定数据分布支持上的三个结构条件，这三个条件是随机化提高稳健性所必需的。此外，我们能够构建几个数据分布，其中(与二进制分类相反)从确定性解决方案切换到随机解决方案显著地降低了最优对抗风险。这些发现突显了随机化在增强多类分类中对抗攻击的稳健性方面所起的关键作用。



## **23. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

利用一对多关系对视觉语言模型进行多模式对抗防御 cs.CV

Under review

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2405.18770v2) [paper-pdf](http://arxiv.org/pdf/2405.18770v2)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. Our experiments show that MAT can effectively be applied to different VL models and tasks to improve adversarial robustness, outperforming previous efforts. Our code will be made public upon acceptance.

摘要: 预先训练好的视觉语言(VL)模型极易受到敌意攻击。然而，现有的防御方法主要集中在图像分类上，忽略了VL任务的两个关键方面：多模式攻击，其中图像和文本都可以被干扰，以及图像和文本的一对多关系，其中一幅图像可以对应于多个文本描述，反之亦然(1：N和N：1)。该工作首次探索了VL任务中针对多模式攻击的防御策略，而以往的VL防御方法侧重于视觉稳健性。我们提出了多模式对抗训练(MAT)，它在训练过程中结合了图像和文本模式的对抗扰动，显著优于现有的单模式防御。此外，我们发现MAT受到VL训练数据中确定性的一对一(1：1)图文对的限制。为了解决这个问题，我们进行了一项关于利用一对多关系来增强健壮性的全面研究，调查了各种增强技术。我们的分析表明，为了更有效地防御，增强的图文对应该很好地对齐、多样化，但又要避免分布偏移--这是以前的研究忽视的条件。我们的实验表明，MAT可以有效地应用于不同的虚拟学习模型和任务，以提高对手的健壮性，表现出优于以往的努力。我们的代码将在接受后公开。



## **24. XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants**

XOXO：针对人工智能编码助理的隐形跨源上下文中毒攻击 cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14281v1) [paper-pdf](http://arxiv.org/pdf/2503.14281v1)

**Authors**: Adam Štorek, Mukur Gupta, Noopur Bhatt, Aditya Gupta, Janie Kim, Prashast Srivastava, Suman Jana

**Abstract**: AI coding assistants are widely used for tasks like code generation, bug detection, and comprehension. These tools now require large and complex contexts, automatically sourced from various origins$\unicode{x2014}$across files, projects, and contributors$\unicode{x2014}$forming part of the prompt fed to underlying LLMs. This automatic context-gathering introduces new vulnerabilities, allowing attackers to subtly poison input to compromise the assistant's outputs, potentially generating vulnerable code, overlooking flaws, or introducing critical errors. We propose a novel attack, Cross-Origin Context Poisoning (XOXO), that is particularly challenging to detect as it relies on adversarial code modifications that are semantically equivalent. Traditional program analysis techniques struggle to identify these correlations since the semantics of the code remain correct, making it appear legitimate. This allows attackers to manipulate code assistants into producing incorrect outputs, including vulnerabilities or backdoors, while shifting the blame to the victim developer or tester. We introduce a novel, task-agnostic black-box attack algorithm GCGS that systematically searches the transformation space using a Cayley Graph, achieving an 83.09% attack success rate on average across five tasks and eleven models, including GPT-4o and Claude 3.5 Sonnet v2 used by many popular AI coding assistants. Furthermore, existing defenses, including adversarial fine-tuning, are ineffective against our attack, underscoring the need for new security measures in LLM-powered coding tools.

摘要: AI编码助手被广泛用于代码生成、错误检测和理解等任务。这些工具现在需要大型而复杂的上下文，这些上下文自动来自不同来源的$\unicode{x2014}$，跨文件、项目和贡献者$\unicode{x2014}$，构成馈送到底层LLM的提示的一部分。这种自动上下文收集引入了新的漏洞，允许攻击者巧妙地毒化输入以危害助手的输出，可能会生成易受攻击的代码、忽略缺陷或引入严重错误。我们提出了一种新的攻击，跨来源上下文中毒(XOXO)，由于它依赖于语义等价的敌意代码修改，因此检测起来特别具有挑战性。传统的程序分析技术很难识别这些相关性，因为代码的语义保持正确，使其看起来是合法的。这允许攻击者操纵代码助手产生不正确的输出，包括漏洞或后门，同时将责任转嫁给受害者开发人员或测试人员。提出了一种新的任务无关的黑盒攻击算法GCGS，该算法使用Cayley图系统地搜索变换空间，在5个任务和11个模型上获得了83.09%的平均攻击成功率，其中包括许多流行的AI编码助手使用的GPT-40和Claude 3.5 Sonnet v2。此外，现有的防御措施，包括对抗性微调，对我们的攻击是无效的，这突显了在LLM支持的编码工具中需要新的安全措施。



## **25. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

TAROT：使用策略优化方法的面向任务的作者混淆 cs.CL

Accepted to the NAACL PrivateNLP 2025 Workshop

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2407.21630v2) [paper-pdf](http://arxiv.org/pdf/2407.21630v2)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduces the accuracy of attackers while preserving utility. We make our code and models publicly available.

摘要: 作者身份混淆旨在通过改变与文本作者相关的写作风格、词汇、句法和其他语言特征来掩盖作者在文本中的身份。这一改变需要平衡隐私和效用。虽然强大的混淆技术可以有效地隐藏作者的身份，但它们往往会降低文本的质量和对预期目的的有用性。相反，保持高实用性往往会提供不充分的隐私，使对手更容易解除作者的匿名。因此，在这两个相互冲突的目标之间实现最佳权衡至关重要。在本文中，我们提出了一种新的无监督作者身份混淆方法--TAROT：基于策略优化的面向任务的作者身份混淆方法，其目标是通过重新生成考虑下游效用的整个文本来优化隐私和效用之间的权衡。我们的方法利用策略优化作为小语言模型上的微调范式，以便通过保留作者身份和下游任务效用来重写文本。我们表明，我们的方法在很大程度上降低了攻击者的准确性，同时保持了实用性。我们公开我们的代码和模型。



## **26. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

针对越狱攻击的多模式大型语言模型对抗训练 cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.04833v2) [paper-pdf](http://arxiv.org/pdf/2503.04833v2)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.

摘要: 多通道大语言模型在跨通道理解和生成任务方面取得了显著进展。然而，它们仍然容易受到越狱攻击，在越狱攻击中，精心设计的扰动绕过安全护栏，引发有害输出。在这篇文章中，我们提出了在MLLM训练阶段为防御越狱攻击而定制的第一个对抗性训练(AT)范例。将传统的AT扩展到这一领域会带来两个关键挑战：有效地调整大量参数和确保对跨多个通道的攻击的健壮性。为了应对这些挑战，我们引入了投影层对抗对手训练(ProEAT)，这是一个端到端的AT框架。ProEAT结合了基于投影仪的对抗性训练体系结构，通过将对抗性训练集中在轻量级投影器层而不是整个模型上，在保持计算可行性的同时有效地处理大规模参数；此外，我们设计了动态权重调整机制，基于任务需求优化损失函数的权重分配，从而简化了调整过程。为了提高防御性能，我们提出了一种跨视觉和文本模式的联合优化策略，确保对来自任何一种模式的越狱攻击具有强大的抵抗力。在三种主流MLLMS上对五种主要的越狱攻击方法进行了广泛的实验，证明了该方法的有效性。ProEAT实现了最先进的防御性能，在文本和图像模式中的表现比现有基线平均高出34%，而干净的准确性仅降低了1%。此外，对真实世界体现的智能系统的评估突出了我们框架的实用适用性，为开发更安全可靠的多式联运系统铺平了道路。



## **27. Survey of Adversarial Robustness in Multimodal Large Language Models**

多模式大型语言模型中的对抗鲁棒性研究 cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](http://arxiv.org/pdf/2503.13962v1)

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.

摘要: 多模式大语言模型(MLLM)通过促进跨不同模式的集成理解，包括文本、图像、视频、音频和语音，在人工智能中表现出出色的性能。然而，它们在实际应用程序中的部署引发了人们对可能危及其安全性和可靠性的对抗性漏洞的严重担忧。与单模模型不同，由于各通道之间的相互依赖关系，最大似然模型面临着独特的挑战，这使得它们容易受到特定通道的威胁和跨通道的对抗性操作。本文综述了MLLMS的对抗稳健性，涵盖了不同的模式。我们首先概述MLLMS和针对每种模式量身定做的对抗性攻击分类。接下来，我们回顾了用于评估MLLMS稳健性的关键数据集和评估指标。在此之后，我们提供了针对不同模式的MLLM的攻击的深入回顾。我们的调查还确定了关键挑战，并提出了前景看好的未来研究方向。



## **28. Make the Most of Everything: Further Considerations on Disrupting Diffusion-based Customization**

充分利用一切：关于破坏基于扩散的定制的进一步思考 cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13945v1) [paper-pdf](http://arxiv.org/pdf/2503.13945v1)

**Authors**: Long Tang, Dengpan Ye, Sirun Chen, Xiuwen Shi, Yunna Lv, Ziyi Liu

**Abstract**: The fine-tuning technique for text-to-image diffusion models facilitates image customization but risks privacy breaches and opinion manipulation. Current research focuses on prompt- or image-level adversarial attacks for anti-customization, yet it overlooks the correlation between these two levels and the relationship between internal modules and inputs. This hinders anti-customization performance in practical threat scenarios. We propose Dual Anti-Diffusion (DADiff), a two-stage adversarial attack targeting diffusion customization, which, for the first time, integrates the adversarial prompt-level attack into the generation process of image-level adversarial examples. In stage 1, we generate prompt-level adversarial vectors to guide the subsequent image-level attack. In stage 2, besides conducting the end-to-end attack on the UNet model, we disrupt its self- and cross-attention modules, aiming to break the correlations between image pixels and align the cross-attention results computed using instance prompts and adversarial prompt vectors within the images. Furthermore, we introduce a local random timestep gradient ensemble strategy, which updates adversarial perturbations by integrating random gradients from multiple segmented timesets. Experimental results on various mainstream facial datasets demonstrate 10%-30% improvements in cross-prompt, keyword mismatch, cross-model, and cross-mechanism anti-customization with DADiff compared to existing methods.

摘要: 文本到图像扩散模型的微调技术为图像定制提供了便利，但存在侵犯隐私和操纵意见的风险。目前的研究侧重于针对反定制的提示级或图像级对抗性攻击，而忽略了这两个级别之间的相关性以及内部模块和输入之间的关系。这阻碍了实际威胁场景中的反定制性能。本文提出了双重反扩散攻击(DADiff)，这是一种针对扩散定制的两阶段对抗性攻击，首次将对抗性提示级攻击融入到图像级对抗性实例的生成过程中。在第一阶段，我们生成提示级对抗向量来指导后续的图像级攻击。在第二阶段中，除了对UNET模型进行端到端攻击外，我们还扰乱了其自我注意和交叉注意模块，旨在打破图像像素之间的相关性，并对齐使用图像中的实例提示和对抗性提示向量计算的交叉注意结果。此外，我们还引入了一种局部随机时间步长梯度集成策略，该策略通过整合来自多个分段时间集的随机梯度来更新对抗性扰动。在各种主流人脸数据集上的实验结果表明，与现有方法相比，DADiff在交叉提示、关键词不匹配、跨模型和跨机制反定制等方面都有10%-30%的改进。



## **29. GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack**

GSBA$^K$：$top$-$K$基于几何分数的黑匣子攻击 cs.CV

This article has been accepted for publication at ICLR 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.12827v2) [paper-pdf](http://arxiv.org/pdf/2503.12827v2)

**Authors**: Md Farhamdur Reza, Richeng Jin, Tianfu Wu, Huaiyu Dai

**Abstract**: Existing score-based adversarial attacks mainly focus on crafting $top$-1 adversarial examples against classifiers with single-label classification. Their attack success rate and query efficiency are often less than satisfactory, particularly under small perturbation requirements; moreover, the vulnerability of classifiers with multi-label learning is yet to be studied. In this paper, we propose a comprehensive surrogate free score-based attack, named \b geometric \b score-based \b black-box \b attack (GSBA$^K$), to craft adversarial examples in an aggressive $top$-$K$ setting for both untargeted and targeted attacks, where the goal is to change the $top$-$K$ predictions of the target classifier. We introduce novel gradient-based methods to find a good initial boundary point to attack. Our iterative method employs novel gradient estimation techniques, particularly effective in $top$-$K$ setting, on the decision boundary to effectively exploit the geometry of the decision boundary. Additionally, GSBA$^K$ can be used to attack against classifiers with $top$-$K$ multi-label learning. Extensive experimental results on ImageNet and PASCAL VOC datasets validate the effectiveness of GSBA$^K$ in crafting $top$-$K$ adversarial examples.

摘要: 现有的基于分数的对抗性攻击主要集中在针对具有单标签分类的分类器构造$top$-1个对抗性实例。它们的攻击成功率和查询效率往往不尽如人意，特别是在小扰动要求下；此外，具有多标签学习的分类器的脆弱性还有待研究。在本文中，我们提出了一种全面的基于分数的无代理攻击，称为基于几何分数的黑盒攻击(GSBA$^K$)，以在攻击性的$top$-$K$环境中为非目标攻击和目标攻击创建对手示例，其中目标是改变目标分类器的$top$-$K$预测。我们引入了新的基于梯度的方法来寻找一个好的初始边界点进行攻击。我们的迭代方法使用了新的梯度估计技术，特别是在决策边界上的$top$-$K$设置上，以有效地利用决策边界的几何形状。此外，GSBA$^K$可用于攻击具有$TOP$-$K$多标签学习的分类器。在ImageNet和Pascal VOC数据集上的大量实验结果验证了GSBA$^K$在构造$TOP$-$K$对抗性实例方面的有效性。



## **30. Securing Virtual Reality Experiences: Unveiling and Tackling Cybersickness Attacks with Explainable AI**

保护虚拟现实体验：利用可解释人工智能揭露和应对网络疾病攻击 cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13419v1) [paper-pdf](http://arxiv.org/pdf/2503.13419v1)

**Authors**: Ripan Kumar Kundu, Matthew Denton, Genova Mongalo, Prasad Calyam, Khaza Anuarul Hoque

**Abstract**: The synergy between virtual reality (VR) and artificial intelligence (AI), specifically deep learning (DL)-based cybersickness detection models, has ushered in unprecedented advancements in immersive experiences by automatically detecting cybersickness severity and adaptively various mitigation techniques, offering a smooth and comfortable VR experience. While this DL-enabled cybersickness detection method provides promising solutions for enhancing user experiences, it also introduces new risks since these models are vulnerable to adversarial attacks; a small perturbation of the input data that is visually undetectable to human observers can fool the cybersickness detection model and trigger unexpected mitigation, thus disrupting user immersive experiences (UIX) and even posing safety risks. In this paper, we present a new type of VR attack, i.e., a cybersickness attack, which successfully stops the triggering of cybersickness mitigation by fooling DL-based cybersickness detection models and dramatically hinders the UIX. Next, we propose a novel explainable artificial intelligence (XAI)-guided cybersickness attack detection framework to detect such attacks in VR to ensure UIX and a comfortable VR experience. We evaluate the proposed attack and the detection framework using two state-of-the-art open-source VR cybersickness datasets: Simulation 2021 and Gameplay dataset. Finally, to verify the effectiveness of our proposed method, we implement the attack and the XAI-based detection using a testbed with a custom-built VR roller coaster simulation with an HTC Vive Pro Eye headset and perform a user study. Our study shows that such an attack can dramatically hinder the UIX. However, our proposed XAI-guided cybersickness attack detection can successfully detect cybersickness attacks and trigger the proper mitigation, effectively reducing VR cybersickness.

摘要: 虚拟现实(VR)和人工智能(AI)之间的协同，特别是基于深度学习(DL)的晕屏检测模型，通过自动检测晕屏的严重程度和自适应的各种缓解技术，带来了沉浸式体验的前所未有的进步，提供了流畅舒适的VR体验。虽然这种支持DL的晕车检测方法为增强用户体验提供了有希望的解决方案，但它也引入了新的风险，因为这些模型容易受到对手攻击；输入数据的微小扰动在视觉上无法被人类观察者检测到，就可以欺骗晕车检测模型并引发意外的缓解，从而扰乱用户沉浸式体验(UIX)，甚至构成安全风险。在本文中，我们提出了一种新型的虚拟现实攻击，即晕屏攻击，它通过愚弄基于DL的晕屏检测模型成功地阻止了晕屏缓解的触发，并显著阻碍了UIX。接下来，我们提出了一种新颖的可解释人工智能(XAI)引导的网络病攻击检测框架来检测VR中的此类攻击，以确保UIX和舒适的VR体验。我们使用两个最先进的开源虚拟现实晕屏数据集：模拟2021和游戏玩法数据集来评估所提出的攻击和检测框架。最后，为了验证我们提出的方法的有效性，我们使用HTC Vive Pro Eye耳机通过定制的VR过山车模拟实验床实现了攻击和基于XAI的检测，并进行了用户研究。我们的研究表明，这样的攻击可以极大地阻碍UIX。然而，我们提出的XAI引导的晕屏攻击检测可以成功检测到晕屏攻击并触发适当的缓解，有效地减少VR晕屏。



## **31. On the Byzantine-Resilience of Distillation-Based Federated Learning**

基于蒸馏的联邦学习的拜占庭弹性 cs.LG

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2402.12265v3) [paper-pdf](http://arxiv.org/pdf/2402.12265v3)

**Authors**: Christophe Roux, Max Zimmer, Sebastian Pokutta

**Abstract**: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and instead communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process. Based on these insights, we introduce two new byzantine attacks and demonstrate their ability to break existing byzantine-resilient methods. Additionally, we propose a novel defence method which enhances the byzantine resilience of KD-based FL algorithms. Finally, we provide a general framework to obfuscate attacks, making them significantly harder to detect, thereby improving their effectiveness. Our findings serve as an important building block in the analysis of byzantine FL, contributing through the development of new attacks and new defence mechanisms, further advancing the robustness of KD-based FL algorithms.

摘要: 基于知识蒸馏(KD)的联合学习(FL)算法因其在隐私、非I.I.D.等方面的良好特性而受到越来越多的关注。数据和通信成本。这些方法不同于传输模型参数，而是通过共享对公共数据集的预测来传递关于学习任务的信息。在这项工作中，我们研究了这些方法在拜占庭环境下的性能，在拜占庭环境中，客户的子集以对抗性的方式行动，旨在扰乱学习过程。我们证明了基于KD的FL算法具有显著的弹性，并分析了拜占庭客户端如何影响学习过程。基于这些见解，我们引入了两个新的拜占庭攻击，并展示了它们打破现有拜占庭弹性方法的能力。此外，我们还提出了一种新的防御方法，增强了基于KD的FL算法的拜占庭抗攻击能力。最后，我们提供了一个通用框架来混淆攻击，使它们更难被检测到，从而提高了它们的有效性。我们的发现是拜占庭FL分析的重要组成部分，通过开发新的攻击和新的防御机制，进一步提高了基于KD的FL算法的健壮性。



## **32. How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark**

我的组织学视觉语言基础模型有多好？整体基准 eess.IV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12990v1) [paper-pdf](http://arxiv.org/pdf/2503.12990v1)

**Authors**: Roba Al Majzoub, Hashmat Malik, Muzammal Naseer, Zaigham Zaheer, Tariq Mahmood, Salman Khan, Fahad Khan

**Abstract**: Recently, histopathology vision-language foundation models (VLMs) have gained popularity due to their enhanced performance and generalizability across different downstream tasks. However, most existing histopathology benchmarks are either unimodal or limited in terms of diversity of clinical tasks, organs, and acquisition instruments, as well as their partial availability to the public due to patient data privacy. As a consequence, there is a lack of comprehensive evaluation of existing histopathology VLMs on a unified benchmark setting that better reflects a wide range of clinical scenarios. To address this gap, we introduce HistoVL, a fully open-source comprehensive benchmark comprising images acquired using up to 11 various acquisition tools that are paired with specifically crafted captions by incorporating class names and diverse pathology descriptions. Our Histo-VL includes 26 organs, 31 cancer types, and a wide variety of tissue obtained from 14 heterogeneous patient cohorts, totaling more than 5 million patches obtained from over 41K WSIs viewed under various magnification levels. We systematically evaluate existing histopathology VLMs on Histo-VL to simulate diverse tasks performed by experts in real-world clinical scenarios. Our analysis reveals interesting findings, including large sensitivity of most existing histopathology VLMs to textual changes with a drop in balanced accuracy of up to 25% in tasks such as Metastasis detection, low robustness to adversarial attacks, as well as improper calibration of models evident through high ECE values and low model prediction confidence, all of which can affect their clinical implementation.

摘要: 最近，组织病理学视觉-语言基础模型(VLM)因其在不同下游任务中增强的性能和普适性而广受欢迎。然而，现有的大多数组织病理学基准要么是单一的，要么在临床任务、器官和采集工具的多样性方面受到限制，而且由于患者数据的隐私，它们对公众的部分可用性。因此，在一个统一的基准设置上缺乏对现有组织病理学VLM的全面评估，以更好地反映广泛的临床情景。为了弥补这一差距，我们引入了HistoVL，这是一个完全开源的综合基准，包括使用多达11种不同的采集工具获取的图像，这些工具通过结合类名和不同的病理描述与专门制作的说明相匹配。我们的HISTO-VL包括26个器官，31种癌症类型，以及从14个不同的患者队列中获得的各种组织，总计超过500万个斑块，这些斑块是在不同放大水平下从超过41K的WSIS中获得的。我们系统地评估HISTO-VL上现有的组织病理学VLM，以模拟真实世界临床场景中专家执行的各种任务。我们的分析揭示了有趣的发现，包括大多数现有的组织病理学VLM对文本变化的高度敏感性，在转移检测等任务中平衡准确率下降高达25%，对对抗性攻击的稳健性低，以及通过高EC值和低模型预测置信度明显地对模型进行不正确的校准，所有这些都可能影响其临床实施。



## **33. Distributed Black-box Attack: Do Not Overestimate Black-box Attacks**

分布式黑匣子攻击：不要高估黑匣子攻击 cs.LG

Accepted by ICLR Workshop, 2025

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2210.16371v5) [paper-pdf](http://arxiv.org/pdf/2210.16371v5)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: As cloud computing becomes pervasive, deep learning models are deployed on cloud servers and then provided as APIs to end users. However, black-box adversarial attacks can fool image classification models without access to model structure and weights. Recent studies have reported attack success rates of over 95% with fewer than 1,000 queries. Then the question arises: whether black-box attacks have become a real threat against cloud APIs? To shed some light on this, our research indicates that black-box attacks are not as effective against cloud APIs as proposed in research papers due to several common mistakes that overestimate the efficiency of black-box attacks. To avoid similar mistakes, we conduct black-box attacks directly on cloud APIs rather than local models.

摘要: 随着云计算变得普遍，深度学习模型被部署在云服务器上，然后作为API提供给最终用户。然而，黑匣子对抗攻击可以在不访问模型结构和权重的情况下欺骗图像分类模型。最近的研究报告称，查询少于1，000个，攻击成功率超过95%。那么问题就来了：黑匣子攻击是否已成为对云API的真正威胁？为了阐明这一点，我们的研究表明，由于存在几个高估黑匣子攻击效率的常见错误，黑匣子攻击对云API的效果并不像研究论文中提出的那样有效。为了避免类似的错误，我们直接对云API而不是本地模型进行黑匣子攻击。



## **34. Algebraic Adversarial Attacks on Explainability Models**

对可解释性模型的代数对抗攻击 cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12683v1) [paper-pdf](http://arxiv.org/pdf/2503.12683v1)

**Authors**: Lachlan Simpson, Federico Costanza, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

**Abstract**: Classical adversarial attacks are phrased as a constrained optimisation problem. Despite the efficacy of a constrained optimisation approach to adversarial attacks, one cannot trace how an adversarial point was generated. In this work, we propose an algebraic approach to adversarial attacks and study the conditions under which one can generate adversarial examples for post-hoc explainability models. Phrasing neural networks in the framework of geometric deep learning, algebraic adversarial attacks are constructed through analysis of the symmetry groups of neural networks. Algebraic adversarial examples provide a mathematically tractable approach to adversarial examples. We validate our approach of algebraic adversarial examples on two well-known and one real-world dataset.

摘要: 经典的对抗攻击被描述为一个约束优化问题。尽管约束优化方法对对抗攻击有效，但人们无法追踪对抗点是如何生成的。在这项工作中，我们提出了一种对抗性攻击的代数方法，并研究了为事后可解释性模型生成对抗性示例的条件。在几何深度学习的框架下对神经网络进行分段，通过分析神经网络的对称群来构建代数对抗攻击。代数对抗性例子为对抗性例子提供了一种数学上易于处理的方法。我们在两个知名数据集和一个现实世界数据集上验证了我们的代数对抗示例方法。



## **35. Provably Reliable Conformal Prediction Sets in the Presence of Data Poisoning**

数据中毒情况下可证明可靠的保形预测集 cs.LG

Accepted at ICLR 2025 (Spotlight)

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2410.09878v4) [paper-pdf](http://arxiv.org/pdf/2410.09878v4)

**Authors**: Yan Scholten, Stephan Günnemann

**Abstract**: Conformal prediction provides model-agnostic and distribution-free uncertainty quantification through prediction sets that are guaranteed to include the ground truth with any user-specified probability. Yet, conformal prediction is not reliable under poisoning attacks where adversaries manipulate both training and calibration data, which can significantly alter prediction sets in practice. As a solution, we propose reliable prediction sets (RPS): the first efficient method for constructing conformal prediction sets with provable reliability guarantees under poisoning. To ensure reliability under training poisoning, we introduce smoothed score functions that reliably aggregate predictions of classifiers trained on distinct partitions of the training data. To ensure reliability under calibration poisoning, we construct multiple prediction sets, each calibrated on distinct subsets of the calibration data. We then aggregate them into a majority prediction set, which includes a class only if it appears in a majority of the individual sets. Both proposed aggregations mitigate the influence of datapoints in the training and calibration data on the final prediction set. We experimentally validate our approach on image classification tasks, achieving strong reliability while maintaining utility and preserving coverage on clean data. Overall, our approach represents an important step towards more trustworthy uncertainty quantification in the presence of data poisoning.

摘要: 保角预测通过预测集提供与模型无关和无分布的不确定性量化，这些预测集保证以任何用户指定的概率包括基本事实。然而，在中毒攻击下，保角预测是不可靠的，其中对手同时操纵训练和校准数据，这在实践中可能会显著改变预测集。作为解决方案，我们提出了可靠预测集(RPS)：在中毒情况下构造具有可证明可靠性保证的共形预测集的第一种有效方法。为了确保在训练中毒情况下的可靠性，我们引入了平滑得分函数，它可靠地聚合了在不同的训练数据分区上训练的分类器的预测。为了确保在校准中毒情况下的可靠性，我们构造了多个预测集，每个预测集都在校准数据的不同子集上进行校准。然后我们将它们聚集到一个多数预测集合中，该集合只包括一个类，当它出现在大多数单独的集合中时。这两种建议的聚合都减轻了训练和校准数据中的数据点对最终预测集的影响。我们在实验上验证了我们的方法在图像分类任务上的有效性，在保持实用性和对干净数据的覆盖率的同时实现了很强的可靠性。总体而言，我们的方法代表着在存在数据中毒的情况下朝着更可信的不确定性量化迈出的重要一步。



## **36. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

注意差距：通过查询更新分析检测正在形成的黑匣子对抗攻击 cs.CR

14 pages

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.02986v3) [paper-pdf](http://arxiv.org/pdf/2503.02986v3)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.

摘要: 对抗性攻击仍然是一个严重的威胁，可能会危及机器学习(ML)模型的完整性。特别是，基于查询的黑盒攻击可以在不访问受害者模型的体系结构的情况下生成恶意噪声，使它们在真实世界的上下文中具有实用性。社区已经提出了几种针对对抗性攻击的防御措施，但都被更先进和适应性更强的攻击策略打破了。在这篇文章中，我们提出了一个框架，它检测是否正在生成对抗性噪声实例。与现有的通过监测输入空间来检测对抗性噪声产生的状态防御方法不同，我们的方法在输入更新相似性空间中学习对抗性模式。事实上，我们提出了一种新的度量，称为Delta相似度(DS)，我们表明它更有效地捕获了对手的行为。我们评估了我们的方法针对8种最先进的攻击，包括自适应攻击，在这些攻击中，对手知道防御并试图逃避检测。我们发现，我们的方法在特异性和敏感性方面都明显比现有的防御方法更稳健。



## **37. GAN-Based Single-Stage Defense for Traffic Sign Classification Under Adversarial Patch Attack**

对抗补丁攻击下基于GAN的交通标志分类单级防御 cs.CV

This work has been submitted to the IEEE Transactions on Intelligent  Transportation Systems (T-ITS) for possible publication

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12567v1) [paper-pdf](http://arxiv.org/pdf/2503.12567v1)

**Authors**: Abyad Enan, Mashrur Chowdhury

**Abstract**: Computer Vision plays a critical role in ensuring the safe navigation of autonomous vehicles (AVs). An AV perception module is responsible for capturing and interpreting the surrounding environment to facilitate safe navigation. This module enables AVs to recognize traffic signs, traffic lights, and various road users. However, the perception module is vulnerable to adversarial attacks, which can compromise their accuracy and reliability. One such attack is the adversarial patch attack (APA), a physical attack in which an adversary strategically places a specially crafted sticker on an object to deceive object classifiers. In APA, an adversarial patch is positioned on a target object, leading the classifier to misidentify it. Such an APA can cause AVs to misclassify traffic signs, leading to catastrophic incidents. To enhance the security of an AV perception system against APAs, this study develops a Generative Adversarial Network (GAN)-based single-stage defense strategy for traffic sign classification. This approach is tailored to defend against APAs on different classes of traffic signs without prior knowledge of a patch's design. This study found this approach to be effective against patches of varying sizes. Our experimental analysis demonstrates that the defense strategy presented in this paper improves the classifier's accuracy under APA conditions by up to 80.8% and enhances overall classification accuracy for all the traffic signs considered in this study by 58%, compared to a classifier without any defense mechanism. Our defense strategy is model-agnostic, making it applicable to any traffic sign classifier, regardless of the underlying classification model.

摘要: 计算机视觉在保证自动驾驶车辆的安全导航中起着至关重要的作用。一个AV感知模块负责捕捉和解释周围环境，以促进安全导航。该模块使自动驾驶系统能够识别交通标志、红绿灯和各种道路使用者。然而，感知模块容易受到敌意攻击，这可能会损害其准确性和可靠性。其中一种攻击是对抗性补丁攻击(APA)，这是一种物理攻击，对手在对象上策略性地放置专门制作的贴纸，以欺骗对象分类器。在APA中，敌意补丁被定位在目标对象上，导致分类器错误地识别它。这样的APA可能会导致AVS对交通标志进行错误分类，从而导致灾难性事件。为了提高反病毒感知系统对抗自动识别系统的安全性，提出了一种基于生成性对抗网络的交通标志分类单级防御策略。这种方法是为防御不同类别交通标志上的APA而量身定做的，而不需要事先知道补丁的设计。这项研究发现，这种方法对不同大小的补丁有效。我们的实验分析表明，与没有任何防御机制的分类器相比，本文提出的防御策略在APA条件下将分类器的准确率提高了80.8%，对本研究中考虑的所有交通标志的整体分类准确率提高了58%。我们的防御策略是模型不可知的，使其适用于任何交通标志分类器，而不考虑底层的分类模型。



## **38. On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis**

尖峰神经网络的隐私风险：成员推断分析 cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2502.13191v2) [paper-pdf](http://arxiv.org/pdf/2502.13191v2)

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at https://anonymous.4open.science/r/MIA_SNN-3610.

摘要: 尖峰神经网络(SNN)在实际应用中因其能量效率和稳健性而受到越来越多的研究，但其隐私风险在很大程度上仍未得到检查。在这项工作中，我们调查了SNN对成员关系推断攻击(MIA)的敏感性--MIA是一种主要的隐私威胁，攻击者试图确定给定样本是否属于训练数据集。虽然以前的工作表明，由于SNN的离散、事件驱动的性质，它可能提供固有的健壮性，但我们发现，它的弹性随着延迟(T)的增加而减弱。此外，我们在黑箱设置下引入了一种输入丢弃策略，显著增强了SNN中的成员关系推理。我们的发现挑战了SNN天生更安全的假设，尽管预计SNN会更好，但我们的结果显示，SNN表现出与人工神经网络(ANN)相当的隐私漏洞。我们的代码可以在https://anonymous.4open.science/r/MIA_SNN-3610.上找到



## **39. Towards Privacy-Preserving Data-Driven Education: The Potential of Federated Learning**

迈向保护隐私的数据驱动教育：联邦学习的潜力 cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.13550v1) [paper-pdf](http://arxiv.org/pdf/2503.13550v1)

**Authors**: Mohammad Khalil, Ronas Shakya, Qinyi Liu

**Abstract**: The increasing adoption of data-driven applications in education such as in learning analytics and AI in education has raised significant privacy and data protection concerns. While these challenges have been widely discussed in previous works, there are still limited practical solutions. Federated learning has recently been discoursed as a promising privacy-preserving technique, yet its application in education remains scarce. This paper presents an experimental evaluation of federated learning for educational data prediction, comparing its performance to traditional non-federated approaches. Our findings indicate that federated learning achieves comparable predictive accuracy. Furthermore, under adversarial attacks, federated learning demonstrates greater resilience compared to non-federated settings. We summarise that our results reinforce the value of federated learning as a potential approach for balancing predictive performance and privacy in educational contexts.

摘要: 教育领域越来越多地采用数据驱动应用程序，例如学习分析和教育领域的人工智能，引发了严重的隐私和数据保护问题。虽然这些挑战在之前的作品中得到了广泛讨论，但实际的解决方案仍然有限。联邦学习最近被认为是一种有前途的隐私保护技术，但其在教育中的应用仍然很少。本文对教育数据预测的联邦学习进行了实验评估，并将其性能与传统非联邦方法进行了比较。我们的研究结果表明，联邦学习可以实现相当的预测准确性。此外，在对抗性攻击下，与非联邦环境相比，联邦学习表现出更大的弹性。我们总结说，我们的结果强化了联邦学习作为在教育环境中平衡预测性能和隐私的潜在方法的价值。



## **40. CARNet: Collaborative Adversarial Resilience for Robust Underwater Image Enhancement and Perception**

CARNet：协作对抗弹性，实现稳健的水下图像增强和感知 cs.CV

13 pages, 13 figures

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2309.01102v2) [paper-pdf](http://arxiv.org/pdf/2309.01102v2)

**Authors**: Zengxi Zhang, Zeru Shi, Zhiying Jiang, Jinyuan Liu

**Abstract**: Due to the uneven absorption of different light wavelengths in aquatic environments, underwater images suffer from low visibility and clear color deviations. With the advancement of autonomous underwater vehicles, extensive research has been conducted on learning-based underwater enhancement algorithms. These works can generate visually pleasing enhanced images and mitigate the adverse effects of degraded images on subsequent perception tasks. However, learning-based methods are susceptible to the inherent fragility of adversarial attacks, causing significant disruption in enhanced results. In this work, we introduce a collaborative adversarial resilience network, dubbed CARNet, for underwater image enhancement and subsequent detection tasks. Concretely, we first introduce an invertible network with strong perturbation-perceptual abilities to isolate attacks from underwater images, preventing interference with visual quality enhancement and perceptual tasks. Furthermore, an attack pattern discriminator is introduced to adaptively identify and eliminate various types of attacks. Additionally, we propose a bilevel attack optimization strategy to heighten the robustness of the network against different types of attacks under the collaborative adversarial training of vision-driven and perception-driven attacks. Extensive experiments demonstrate that the proposed method outputs visually appealing enhancement images and performs an average 6.71% higher detection mAP than state-of-the-art methods.

摘要: 由于水环境对不同波长光的吸收不均匀，水下图像的能见度较低，颜色偏差较明显。随着自主水下机器人的发展，基于学习的水下增强算法得到了广泛的研究。这些工作可以产生视觉上令人愉悦的增强图像，并缓解退化图像对后续感知任务的不利影响。然而，基于学习的方法容易受到对抗性攻击固有的脆弱性的影响，从而对增强的结果造成重大破坏。在这项工作中，我们介绍了一个协作的对抗性网络，称为CARNET，用于水下图像增强和后续检测任务。具体地说，我们首先引入了一个具有很强扰动感知能力的可逆网络，将攻击从水下图像中分离出来，防止了对视觉质量增强和感知任务的干扰。此外，还引入了攻击模式鉴别器来自适应地识别和消除各种类型的攻击。此外，在视觉驱动攻击和感知驱动攻击的协同对抗训练下，我们提出了一种双层攻击优化策略，以提高网络对不同类型攻击的鲁棒性。大量实验表明，该方法输出的增强图像视觉效果良好，检测结果比现有方法平均提高6.71%。



## **41. Augmented Adversarial Trigger Learning**

增强对抗触发学习 cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12339v1) [paper-pdf](http://arxiv.org/pdf/2503.12339v1)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs.

摘要: 基于梯度优化的对抗性攻击方法自动学习对抗性触发器，以生成越狱提示或泄漏系统提示。在这项工作中，我们更深入地研究了对抗性触发学习的优化目标，并提出了ATLA：带有扩充目标的对抗性触发学习。ATLA将先前研究使用的负对数似然损失改进为加权损失公式，该公式鼓励学习的对抗性触发器更多地针对响应格式令牌进行优化。这使ATLA能够仅从一个查询-响应对学习敌意触发器，并且学习的触发器很好地推广到其他类似的查询。我们进一步设计了一种变体，以增加辅助损耗来增强触发优化，从而抑制回避响应。我们展示了如何使用ATLA来学习敌意后缀，越狱LLM和提取隐藏的系统提示。经验证明，ATLA始终优于当前最先进的技术，在攻击中取得了近100%的成功，所需查询减少了80%。ATLA学习到的越狱后缀对看不见的查询表现出高度的概括性，并很好地转移到新的LLM。



## **42. Training-Free Mitigation of Adversarial Attacks on Deep Learning-Based MRI Reconstruction**

基于深度学习的MRI重建的对抗攻击的免培训缓解 cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2501.01908v2) [paper-pdf](http://arxiv.org/pdf/2501.01908v2)

**Authors**: Mahdi Saberi, Chi Zhang, Mehmet Akcakaya

**Abstract**: Deep learning (DL) methods, especially those based on physics-driven DL, have become the state-of-the-art for reconstructing sub-sampled magnetic resonance imaging (MRI) data. However, studies have shown that these methods are susceptible to small adversarial input perturbations, or attacks, resulting in major distortions in the output images. Various strategies have been proposed to reduce the effects of these attacks, but they require retraining and may lower reconstruction quality for non-perturbed/clean inputs. In this work, we propose a novel approach for mitigating adversarial attacks on MRI reconstruction models without any retraining. Our framework is based on the idea of cyclic measurement consistency. The output of the model is mapped to another set of MRI measurements for a different sub-sampling pattern, and this synthesized data is reconstructed with the same model. Intuitively, without an attack, the second reconstruction is expected to be consistent with the first, while with an attack, disruptions are present. A novel objective function is devised based on this idea, which is minimized within a small ball around the attack input for mitigation. Experimental results show that our method substantially reduces the impact of adversarial perturbations across different datasets, attack types/strengths and PD-DL networks, and qualitatively and quantitatively outperforms conventional mitigation methods that involve retraining. Finally, we extend our mitigation method to two important practical scenarios: a blind setup, where the attack strength or algorithm is not known to the end user; and an adaptive attack setup, where the attacker has full knowledge of the defense strategy. Our approach remains effective in both cases.

摘要: 深度学习方法，特别是基于物理驱动的深度学习方法，已经成为重建亚采样磁共振成像(MRI)数据的最新方法。然而，研究表明，这些方法容易受到小的对抗性输入扰动或攻击，导致输出图像的严重失真。已经提出了各种策略来减少这些攻击的影响，但它们需要重新培训，并且可能会降低非扰动/干净输入的重建质量。在这项工作中，我们提出了一种新的方法来缓解对MRI重建模型的敌意攻击，而不需要任何重新训练。我们的框架是基于循环测量一致性的思想。该模型的输出被映射到用于不同子采样模式的另一组MRI测量，并且利用相同的模型重建该合成数据。直观地说，在没有攻击的情况下，第二次重建预计与第一次一致，而在攻击时，会出现中断。基于这一思想，设计了一种新的目标函数，该函数在攻击输入周围的小球内最小化以减少攻击。实验结果表明，我们的方法大大降低了不同数据集、攻击类型/强度和PD-DL网络之间的对抗性扰动的影响，并且在定性和定量上都优于传统的涉及再训练的缓解方法。最后，我们将我们的缓解方法扩展到两个重要的实际场景：盲目设置，其中最终用户不知道攻击强度或算法；以及自适应攻击设置，其中攻击者完全了解防御策略。我们的方法在这两种情况下仍然有效。



## **43. Multi-Agent Systems Execute Arbitrary Malicious Code**

多代理系统执行任意恶意代码 cs.CR

30 pages, 5 figures, 8 tables

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2503.12188v1) [paper-pdf](http://arxiv.org/pdf/2503.12188v1)

**Authors**: Harold Triedman, Rishi Jha, Vitaly Shmatikov

**Abstract**: Multi-agent systems coordinate LLM-based agents to perform tasks on users' behalf. In real-world applications, multi-agent systems will inevitably interact with untrusted inputs, such as malicious Web content, files, email attachments, etc.   Using several recently proposed multi-agent frameworks as concrete examples, we demonstrate that adversarial content can hijack control and communication within the system to invoke unsafe agents and functionalities. This results in a complete security breach, up to execution of arbitrary malicious code on the user's device and/or exfiltration of sensitive data from the user's containerized environment. We show that control-flow hijacking attacks succeed even if the individual agents are not susceptible to direct or indirect prompt injection, and even if they refuse to perform harmful actions.

摘要: 多代理系统协调基于LLM的代理代表用户执行任务。在现实世界的应用程序中，多代理系统将不可避免地与不受信任的输入进行交互，例如恶意Web内容、文件、电子邮件附件等。   使用最近提出的几个多代理框架作为具体示例，我们证明对抗性内容可以劫持系统内的控制和通信，以调用不安全的代理和功能。这会导致完全的安全漏洞，甚至在用户设备上执行任意恶意代码和/或从用户的容器化环境中泄露敏感数据。我们表明，即使各个代理不容易受到直接或间接的即时注入，并且即使他们拒绝执行有害动作，控制流劫持攻击也会成功。



## **44. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2407.20361v3) [paper-pdf](http://arxiv.org/pdf/2407.20361v3)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models -- Stack model, VisualPhishNet, and Phishpedia -- against PhishOracle-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a multimodal large language model (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether PhishOracle-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources will be made publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的三种特定任务的模型--Stack模型、VisualPhishNet和Phishpedia--对PhishOracle生成的敌意钓鱼网页的稳健性，并观察到它们的检测率显著下降。相比之下，基于多模式大语言模型(MLLM)的网络钓鱼检测器对这些敌意攻击表现出更强的稳健性，但仍然容易被规避。我们的发现突出了网络钓鱼检测模型对对手攻击的脆弱性，强调了需要更强大的检测方法。此外，我们还进行了一项用户研究，以评估PhishOracle生成的敌意钓鱼网页是否可以欺骗用户。结果表明，许多钓鱼网页不仅规避了现有的检测模型，而且还规避了用户。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都将在GitHub上公开提供。



## **45. Robust Dataset Distillation by Matching Adversarial Trajectories**

通过匹配对抗轨迹进行稳健的数据集提取 cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2503.12069v1) [paper-pdf](http://arxiv.org/pdf/2503.12069v1)

**Authors**: Wei Lai, Tianyu Ding, ren dongdong, Lei Wang, Jing Huo, Yang Gao, Wenbin Li

**Abstract**: Dataset distillation synthesizes compact datasets that enable models to achieve performance comparable to training on the original large-scale datasets. However, existing distillation methods overlook the robustness of the model, resulting in models that are vulnerable to adversarial attacks when trained on distilled data. To address this limitation, we introduce the task of ``robust dataset distillation", a novel paradigm that embeds adversarial robustness into the synthetic datasets during the distillation process. We propose Matching Adversarial Trajectories (MAT), a method that integrates adversarial training into trajectory-based dataset distillation. MAT incorporates adversarial samples during trajectory generation to obtain robust training trajectories, which are then used to guide the distillation process. As experimentally demonstrated, even through natural training on our distilled dataset, models can achieve enhanced adversarial robustness while maintaining competitive accuracy compared to existing distillation methods. Our work highlights robust dataset distillation as a new and important research direction and provides a strong baseline for future research to bridge the gap between efficient training and adversarial robustness.

摘要: 数据集精馏合成了紧凑的数据集，使模型能够获得与原始大规模数据集上的训练相当的性能。然而，现有的蒸馏方法忽略了模型的稳健性，导致在对蒸馏数据进行训练时，模型容易受到敌意攻击。为了解决这一局限性，我们引入了“稳健数据集精馏”的任务，这是一种在精馏过程中将对抗性健壮性嵌入到合成数据集中的新范例。我们提出了匹配对抗性轨迹(MAT)，一种将对抗性训练与基于轨迹的数据集提取相结合的方法。MAT在轨迹生成过程中加入对抗性样本，以获得稳健的训练轨迹，然后使用这些轨迹来指导蒸馏过程。正如实验所证明的，即使通过在我们的蒸馏数据集上的自然训练，模型也可以实现增强的对抗性稳健性，同时与现有的蒸馏方法相比保持竞争的准确性。我们的工作突出了稳健数据集蒸馏作为一个新的重要研究方向，并为未来的研究提供了一个强有力的基线，以弥合有效训练和对手稳健性之间的差距。



## **46. On Minimizing Adversarial Counterfactual Error in Adversarial RL**

对抗性RL中对抗性反事实错误的最小化 cs.LG

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2406.04724v3) [paper-pdf](http://arxiv.org/pdf/2406.04724v3)

**Authors**: Roman Belaire, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Deep Reinforcement Learning (DRL) policies are highly susceptible to adversarial noise in observations, which poses significant risks in safety-critical scenarios. The challenge inherent to adversarial perturbations is that by altering the information observed by the agent, the state becomes only partially observable. Existing approaches address this by either enforcing consistent actions across nearby states or maximizing the worst-case value within adversarially perturbed observations. However, the former suffers from performance degradation when attacks succeed, while the latter tends to be overly conservative, leading to suboptimal performance in benign settings. We hypothesize that these limitations stem from their failing to account for partial observability directly. To this end, we introduce a novel objective called Adversarial Counterfactual Error (ACoE), defined on the beliefs about the true state and balancing value optimization with robustness. To make ACoE scalable in model-free settings, we propose the theoretically-grounded surrogate objective Cumulative-ACoE (C-ACoE). Our empirical evaluations on standard benchmarks (MuJoCo, Atari, and Highway) demonstrate that our method significantly outperforms current state-of-the-art approaches for addressing adversarial RL challenges, offering a promising direction for improving robustness in DRL under adversarial conditions. Our code is available at https://github.com/romanbelaire/acoe-robust-rl.

摘要: 深度强化学习(DRL)策略很容易受到观测中的对抗性噪声的影响，这在安全关键的场景中会带来巨大的风险。对抗性扰动固有的挑战是，通过改变代理人观察到的信息，状态只变得部分可观察到。现有的方法要么在附近的州强制执行一致的行动，要么在相反的扰动观测中最大化最坏情况的值来解决这个问题。然而，当攻击成功时，前者的性能会下降，而后者往往过于保守，导致在良性设置下的性能不佳。我们假设，这些限制源于它们未能直接解释部分可观测性。为此，我们引入了一个新的目标，称为对抗性反事实错误(ACoE)，它定义在关于真实状态的信念和价值优化与稳健性之间的平衡。为了使ACoE在无模型环境下具有可扩展性，我们提出了基于理论的代理目标累积ACoE(C-ACoE)。我们在标准基准(MuJoCo、Atari和Road)上的经验评估表明，我们的方法显著优于当前最先进的方法来应对对抗性的RL挑战，为提高对抗性条件下的DRL的健壮性提供了一个有希望的方向。我们的代码可以在https://github.com/romanbelaire/acoe-robust-rl.上找到



## **47. Robust and Efficient Adversarial Defense in SNNs via Image Purification and Joint Detection**

通过图像净化和联合检测在SNN中实现稳健有效的对抗防御 cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2404.17092v2) [paper-pdf](http://arxiv.org/pdf/2404.17092v2)

**Authors**: Weiran Chen, Qi Xu

**Abstract**: Spiking Neural Networks (SNNs) aim to bridge the gap between neuroscience and machine learning by emulating the structure of the human nervous system. However, like convolutional neural networks, SNNs are vulnerable to adversarial attacks. To tackle the challenge, we propose a biologically inspired methodology to enhance the robustness of SNNs, drawing insights from the visual masking effect and filtering theory. First, an end-to-end SNN-based image purification model is proposed to defend against adversarial attacks, including a noise extraction network and a non-blind denoising network. The former network extracts noise features from noisy images, while the latter component employs a residual U-Net structure to reconstruct high-quality noisy images and generate clean images. Simultaneously, a multi-level firing SNN based on Squeeze-and-Excitation Network is introduced to improve the robustness of the classifier. Crucially, the proposed image purification network serves as a pre-processing module, avoiding modifications to classifiers. Unlike adversarial training, our method is highly flexible and can be seamlessly integrated with other defense strategies. Experimental results on various datasets demonstrate that the proposed methodology outperforms state-of-the-art baselines in terms of defense effectiveness, training time, and resource consumption.

摘要: 尖峰神经网络(SNN)旨在通过模拟人类神经系统的结构来弥合神经科学和机器学习之间的差距。然而，与卷积神经网络一样，SNN也容易受到敌意攻击。为了应对这一挑战，我们借鉴视觉掩蔽效应和过滤理论，提出了一种受生物启发的方法来增强SNN的稳健性。首先，提出了一种端到端的SNN图像净化模型，该模型包括噪声提取网络和非盲去噪网络。前者从含噪图像中提取噪声特征，后者利用残差U网结构重建高质量的含噪图像并生成清晰的图像。同时，为了提高分类器的鲁棒性，引入了一种基于挤压激励网络的多级激发SNN。最重要的是，提出的图像净化网络作为一个预处理模块，避免了对分类器的修改。与对抗性训练不同，我们的方法高度灵活，可以与其他防御策略无缝集成。在不同数据集上的实验结果表明，该方法在防御效果、训练时间和资源消耗方面都优于最新的基线。



## **48. A Framework for Evaluating Emerging Cyberattack Capabilities of AI**

评估人工智能新兴网络攻击能力的框架 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11917v1) [paper-pdf](http://arxiv.org/pdf/2503.11917v1)

**Authors**: Mikel Rodriguez, Raluca Ada Popa, Four Flynn, Lihao Liang, Allan Dafoe, Anna Wang

**Abstract**: As frontier models become more capable, the community has attempted to evaluate their ability to enable cyberattacks. Performing a comprehensive evaluation and prioritizing defenses are crucial tasks in preparing for AGI safely. However, current cyber evaluation efforts are ad-hoc, with no systematic reasoning about the various phases of attacks, and do not provide a steer on how to use targeted defenses. In this work, we propose a novel approach to AI cyber capability evaluation that (1) examines the end-to-end attack chain, (2) helps to identify gaps in the evaluation of AI threats, and (3) helps defenders prioritize targeted mitigations and conduct AI-enabled adversary emulation to support red teaming. To achieve these goals, we propose adapting existing cyberattack chain frameworks to AI systems. We analyze over 12,000 instances of real-world attempts to use AI in cyberattacks catalogued by Google's Threat Intelligence Group. Using this analysis, we curate a representative collection of seven cyberattack chain archetypes and conduct a bottleneck analysis to identify areas of potential AI-driven cost disruption. Our evaluation benchmark consists of 50 new challenges spanning different phases of cyberattacks. Based on this, we devise targeted cybersecurity model evaluations, report on the potential for AI to amplify offensive cyber capabilities across specific attack phases, and conclude with recommendations on prioritizing defenses. In all, we consider this to be the most comprehensive AI cyber risk evaluation framework published so far.

摘要: 随着前沿模型变得更有能力，该社区试图评估他们启用网络攻击的能力。进行全面的评估和确定防御的优先顺序是安全准备AGI的关键任务。然而，目前的网络评估工作是临时的，没有关于攻击的各个阶段的系统推理，也没有为如何使用定向防御提供指导。在这项工作中，我们提出了一种新的人工智能网络能力评估方法，该方法(1)检查端到端攻击链，(2)帮助识别人工智能威胁评估中的差距，(3)帮助防御者确定目标缓解的优先顺序，并进行支持红色团队的人工智能启用的对手仿真。为了实现这些目标，我们建议将现有的网络攻击链框架应用于人工智能系统。我们分析了谷歌威胁情报集团编目的超过12,000起试图在网络攻击中使用人工智能的真实世界实例。利用这一分析，我们收集了具有代表性的七个网络攻击链原型，并进行了瓶颈分析，以确定潜在的人工智能驱动的成本中断领域。我们的评估基准包括50个新挑战，涵盖网络攻击的不同阶段。在此基础上，我们设计了有针对性的网络安全模型评估，报告了人工智能在特定攻击阶段放大进攻性网络能力的潜力，并对防御的优先顺序提出了建议。总而言之，我们认为这是迄今为止发布的最全面的人工智能网络风险评估框架。



## **49. Order Fairness Evaluation of DAG-based ledgers**

基于DAB的分类帐的订单公平性评估 cs.CR

17 double-column pages with 9 pages dedicated to references and  appendices, 22 figures, 13 of which are in the appendices

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2502.17270v2) [paper-pdf](http://arxiv.org/pdf/2502.17270v2)

**Authors**: Erwan Mahe, Sara Tucci-Piergiovanni

**Abstract**: Order fairness in distributed ledgers refers to properties that relate the order in which transactions are sent or received to the order in which they are eventually finalized, i.e., totally ordered. The study of such properties is relatively new and has been especially stimulated by the rise of Maximal Extractable Value (MEV) attacks in blockchain environments. Indeed, in many classical blockchain protocols, leaders are responsible for selecting the transactions to be included in blocks, which creates a clear vulnerability and opportunity for transaction order manipulation.   Unlike blockchains, DAG-based ledgers allow participants in the network to independently propose blocks, which are then arranged as vertices of a directed acyclic graph. Interestingly, leaders in DAG-based ledgers are elected only after the fact, once transactions are already part of the graph, to determine their total order. In other words, transactions are not chosen by single leaders; instead, they are collectively validated by the nodes, and leaders are only elected to establish an ordering. This approach intuitively reduces the risk of transaction manipulation and enhances fairness.   In this paper, we aim to quantify the capability of DAG-based ledgers to achieve order fairness. To this end, we define new variants of order fairness adapted to DAG-based ledgers and evaluate the impact of an adversary capable of compromising a limited number of nodes (below the one-third threshold) to reorder transactions. We analyze how often our order fairness properties are violated under different network conditions and parameterizations of the DAG algorithm, depending on the adversary's power.   Our study shows that DAG-based ledgers are still vulnerable to reordering attacks, as an adversary can coordinate a minority of Byzantine nodes to manipulate the DAG's structure.

摘要: 分布式分类账中的顺序公平性是指将发送或接收交易的顺序与最终确定的顺序(即完全有序)联系起来的属性。对这类属性的研究相对较新，尤其是区块链环境中最大可提取价值(MEV)攻击的兴起。事实上，在许多经典的区块链协议中，领导者负责选择要包含在区块中的交易，这为交易顺序操纵创造了明显的漏洞和机会。与区块链不同，基于DAG的分类账允许网络中的参与者独立提出区块，然后将这些区块排列为有向无环图的顶点。有趣的是，只有在交易已经成为图表的一部分后，才会在基于DAG的分类账中选出领导人，以确定其总顺序。换句话说，事务不是由单个领导者选择的；相反，它们由节点集体验证，而领导者只被选举来建立顺序。这种方法直观地降低了交易操纵的风险，提高了公平性。在本文中，我们的目标是量化基于DAG的分类帐实现顺序公平的能力。为此，我们定义了适用于基于DAG的分类账的顺序公平性的新变体，并评估了攻击者能够危害有限数量的节点(低于三分之一的阈值)来重新排序事务的影响。我们分析了在不同的网络条件和DAG算法的参数设置下，我们的顺序公平性被违反的频率，这取决于对手的力量。我们的研究表明，基于DAG的分类账仍然容易受到重新排序攻击，因为对手可以协调少数拜占庭节点来操纵DAG的结构。



## **50. Enhancing Resiliency of Sketch-based Security via LSB Sharing-based Dynamic Late Merging**

通过基于TSB共享的动态后期合并增强基于草图的安全性 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11777v1) [paper-pdf](http://arxiv.org/pdf/2503.11777v1)

**Authors**: Seungsam Yang, Seyed Mohammad Mehdi Mirnajafizadeh, Sian Kim, Rhongho Jang, DaeHun Nyang

**Abstract**: With the exponentially growing Internet traffic, sketch data structure with a probabilistic algorithm has been expected to be an alternative solution for non-compromised (non-selective) security monitoring. While facilitating counting within a confined memory space, the sketch's memory efficiency and accuracy were further pushed to their limit through finer-grained and dynamic control of constrained memory space to adapt to the data stream's inherent skewness (i.e., Zipf distribution), namely small counters with extensions. In this paper, we unveil a vulnerable factor of the small counter design by introducing a new sketch-oriented attack, which threatens a stream of state-of-the-art sketches and their security applications. With the root cause analyses, we propose Siamese Counter with enhanced adversarial resiliency and verified feasibility with extensive experimental and theoretical analyses. Under a sketch pollution attack, Siamese Counter delivers 47% accurate results than a state-of-the-art scheme, and demonstrates up to 82% more accurate estimation under normal measurement scenarios.

摘要: 随着互联网流量的指数级增长，使用概率算法的草图数据结构有望成为非妥协(非选择性)安全监控的替代方案。在便于在有限的内存空间内计数的同时，通过对受限的内存空间进行更细粒度的动态控制来适应数据流固有的偏斜性(即Zipf分布)，即带有扩展的小计数器，进一步将草图的内存效率和精度推到了极限。在本文中，我们通过引入一种新的面向草图的攻击来揭示小计数器设计的一个易受攻击的因素，该攻击威胁到了一系列最新的草图及其安全应用。通过对根本原因的分析，我们提出了增强对抗能力的暹罗对抗，并通过大量的实验和理论分析验证了其可行性。在素描污染攻击下，暹罗计数器比最先进的方案提供47%的准确结果，在正常测量场景下的精确度高达82%。



