# Latest Large Language Model Attack Papers
**update at 2025-09-15 14:33:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. URL2Graph++: Unified Semantic-Structural-Character Learning for Malicious URL Detection**

URL 2Shape ++：用于恶意URL检测的统一语义-结构-字符学习 cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10287v1) [paper-pdf](http://arxiv.org/pdf/2509.10287v1)

**Authors**: Ye Tian, Yifan Jia, Yanbin Wang, Jianguo Sun, Zhiquan Liu, Xiaowen Ling

**Abstract**: Malicious URL detection remains a major challenge in cybersecurity, primarily due to two factors: (1) the exponential growth of the Internet has led to an immense diversity of URLs, making generalized detection increasingly difficult; and (2) attackers are increasingly employing sophisticated obfuscation techniques to evade detection. We advocate that addressing these challenges fundamentally requires: (1) obtaining semantic understanding to improve generalization across vast and diverse URL sets, and (2) accurately modeling contextual relationships within the structural composition of URLs. In this paper, we propose a novel malicious URL detection method combining multi-granularity graph learning with semantic embedding to jointly capture semantic, character-level, and structural features for robust URL analysis. To model internal dependencies within URLs, we first construct dual-granularity URL graphs at both subword and character levels, where nodes represent URL tokens/characters and edges encode co-occurrence relationships. To obtain fine-grained embeddings, we initialize node representations using a character-level convolutional network. The two graphs are then processed through jointly trained Graph Convolutional Networks to learn consistent graph-level representations, enabling the model to capture complementary structural features that reflect co-occurrence patterns and character-level dependencies. Furthermore, we employ BERT to derive semantic representations of URLs for semantically aware understanding. Finally, we introduce a gated dynamic fusion network to combine the semantically enriched BERT representations with the jointly optimized graph vectors, further enhancing detection performance. We extensively evaluate our method across multiple challenging dimensions. Results show our method exceeds SOTA performance, including against large language models.

摘要: 恶意URL检测仍然是网络安全的一个重大挑战，主要原因是两个因素：（1）互联网的指数级增长导致URL的多样性，使得广义检测变得越来越困难;（2）攻击者越来越多地使用复杂的混淆技术来逃避检测。我们主张解决这些挑战从根本上来说需要：（1）获得语义理解，以改善庞大且多样化的URL集的概括性，（2）准确地建模URL结构组成中的上下文关系。本文提出了一种新型的恶意URL检测方法，该方法将多粒度图学习与语义嵌入相结合，联合捕获语义、字符级和结构特征，以进行稳健的URL分析。为了对URL内的内部依赖关系进行建模，我们首先在子字和字符级别上构建双粒度URL图，其中节点表示URL令牌/字符，边编码共生关系。为了获得细粒度嵌入，我们使用字符级卷积网络初始化节点表示。然后通过联合训练的图卷积网络处理这两个图，以学习一致的图级表示，使模型能够捕获反映共生模式和字符级依赖关系的互补结构特征。此外，我们使用BERT来推导URL的语义表示，以进行语义感知的理解。最后，我们引入门控动态融合网络，将语义丰富的BERT表示与联合优化的图载体相结合，进一步提高检测性能。我们在多个具有挑战性的维度上广泛评估我们的方法。结果表明，我们的方法超过了SOTA性能，包括对大型语言模型。



## **2. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

在岩石和困难之间：利用道德推理越狱法学硕士 cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.05367v2) [paper-pdf](http://arxiv.org/pdf/2509.05367v2)

**Authors**: Shei Pern Chua, Zhen Leng Thai, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.

摘要: 大型语言模型（LLM）已经进行了安全调整工作，以减轻有害输出。然而，随着LLM的推理变得更加复杂，他们的智能可能会带来新的安全风险。虽然传统的越狱攻击依赖于单步攻击，但动态适应上下文的多回合越狱策略仍然没有得到充分的研究。在这项工作中，我们引入了TRAL（交互式攻击逻辑的电车问题推理），这是一个利用LLM道德推理来绕过其保障措施的框架。TRAL将对抗目标嵌入以电车问题为模型的道德困境中。TriAL展示了开放和封闭源模型的高越狱成功率。我们的研究结果强调了人工智能安全性的一个根本限制：随着模型获得高级推理能力，它们的对齐性质可能会无意中允许更多隐蔽的安全漏洞被利用。TRAL提出了重新评估安全一致监督策略的迫切需要，因为当前的保障措施可能不足以抵御上下文感知的对抗攻击。



## **3. Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective**

LLM授权的推荐系统的隐私风险：倒置攻击的角度 cs.IR

Accepted at ACM RecSys 2025 (10 pages, 4 figures)

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2508.03703v2) [paper-pdf](http://arxiv.org/pdf/2508.03703v2)

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.

摘要: 大语言模型（LLM）支持的推荐范式被提出来解决传统推荐系统的局限性，传统推荐系统通常难以处理冷启动用户或具有新ID的项目。尽管有效，这项研究发现LLM授权的推荐系统很容易受到重建攻击，这些攻击可能会暴露系统和用户隐私。为了研究这种威胁，我们对针对LLM授权的推荐系统的倒置攻击进行了首次系统性研究，其中对手试图通过利用推荐模型的输出日志来重建包含个人偏好、交互历史和人口统计属性的原始提示。我们重现vec 2text框架并使用我们提出的名为相似性引导细化的方法对其进行优化，从而能够从模型生成的日志中更准确地重建文本提示。跨两个领域（电影和书籍）的广泛实验和两个代表性的基于LLM的推荐模型证明我们的方法实现了高保真重建。具体来说，我们可以恢复近65%的用户交互项目，并在87%的情况下正确推断年龄和性别。实验还表明，隐私泄露在很大程度上对受害者模型的性能不敏感，但高度依赖于域一致性和提示复杂性。这些发现暴露了LLM授权的推荐系统中的关键隐私漏洞。



## **4. When Your Reviewer is an LLM: Biases, Divergence, and Prompt Injection Risks in Peer Review**

当您的评审员是法学硕士时：同行评审中的偏见、分歧和及时注入风险 cs.CY

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.09912v1) [paper-pdf](http://arxiv.org/pdf/2509.09912v1)

**Authors**: Changjia Zhu, Junjie Xiong, Renkai Ma, Zhicong Lu, Yao Liu, Lingyao Li

**Abstract**: Peer review is the cornerstone of academic publishing, yet the process is increasingly strained by rising submission volumes, reviewer overload, and expertise mismatches. Large language models (LLMs) are now being used as "reviewer aids," raising concerns about their fairness, consistency, and robustness against indirect prompt injection attacks. This paper presents a systematic evaluation of LLMs as academic reviewers. Using a curated dataset of 1,441 papers from ICLR 2023 and NeurIPS 2022, we evaluate GPT-5-mini against human reviewers across ratings, strengths, and weaknesses. The evaluation employs structured prompting with reference paper calibration, topic modeling, and similarity analysis to compare review content. We further embed covert instructions into PDF submissions to assess LLMs' susceptibility to prompt injection. Our findings show that LLMs consistently inflate ratings for weaker papers while aligning more closely with human judgments on stronger contributions. Moreover, while overarching malicious prompts induce only minor shifts in topical focus, explicitly field-specific instructions successfully manipulate specific aspects of LLM-generated reviews. This study underscores both the promises and perils of integrating LLMs into peer review and points to the importance of designing safeguards that ensure integrity and trust in future review processes.

摘要: 同行评审是学术出版的基石，但由于提交量不断增加、评审员超载和专业知识不匹配，这一过程变得越来越紧张。大型语言模型（LLM）现在被用作“审阅者辅助工具”，这引发了人们对其公平性、一致性和针对间接提示注入攻击的稳健性的担忧。本文对法学硕士作为学术评审员进行了系统评估。使用ICLR 2023和NeurIPS 2022的1，441篇论文的精心策划数据集，我们与人类评论员在评级、优势和劣势方面评估GPT-5-mini。评估采用结构化提示、参考论文校准、主题建模和相似性分析来比较评论内容。我们进一步将秘密指令嵌入到PDF提交中，以评估LLM对即时注射的敏感性。我们的研究结果表明，LLM不断夸大较弱论文的评级，同时更接近人类对较强贡献的判断。此外，虽然总体恶意提示只会导致话题焦点的微小转变，但明确的特定领域指令成功地操纵了LLM生成的评论的特定方面。这项研究强调了将法学硕士纳入同行评审的承诺和危险，并指出了设计确保未来评审流程完整性和信任的保障措施的重要性。



## **5. Advancing Security with Digital Twins: A Comprehensive Survey**

通过数字双胞胎提高安全性：全面调查 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2505.17310v2) [paper-pdf](http://arxiv.org/pdf/2505.17310v2)

**Authors**: Blessing Airehenbuwa, Touseef Hasan, Souvika Sarkar, Ujjwal Guin

**Abstract**: The proliferation of electronic devices has greatly transformed every aspect of human life, such as communication, healthcare, transportation, and energy. Unfortunately, the global electronics supply chain is vulnerable to various attacks, including piracy of intellectual properties, tampering, counterfeiting, information leakage, side-channel, and fault injection attacks, due to the complex nature of electronic products and vulnerabilities present in them. Although numerous solutions have been proposed to address these threats, significant gaps remain, particularly in providing scalable and comprehensive protection against emerging attacks. Digital twin, a dynamic virtual replica of a physical system, has emerged as a promising solution to address these issues by providing backward traceability, end-to-end visibility, and continuous verification of component integrity and behavior. In this paper, we comprehensively present the latest digital twin-based security implementations, including their role in cyber-physical systems, Internet of Things, cryptographic systems, detection of counterfeit electronics, intrusion detection, fault injection, and side-channel leakage. This work considers these critical security use cases within a single study to offer researchers and practitioners a unified reference for securing hardware with digital twins. The paper also explores the integration of large language models with digital twins for enhanced security and discusses current challenges, solutions, and future research directions.

摘要: 电子设备的激增极大地改变了人类生活的各个方面，例如通信、医疗保健、交通和能源。不幸的是，由于电子产品的复杂性及其存在的漏洞，全球电子供应链很容易受到各种攻击，包括知识产权盗版、篡改、假冒、信息泄露、侧通道和故障注入攻击。尽管已经提出了许多解决方案来解决这些威胁，但仍然存在巨大差距，特别是在针对新出现的攻击提供可扩展和全面的保护方面。Digital twin是物理系统的动态虚拟副本，通过提供向后可追溯性、端到端可见性以及组件完整性和行为的持续验证，已成为解决这些问题的一种有前途的解决方案。在本文中，我们全面介绍了最新的基于数字孪生的安全实现，包括它们在网络物理系统、物联网、加密系统、假冒电子产品检测、入侵检测、故障注入和侧通道泄漏中的作用。这项工作在一项研究中考虑了这些关键的安全用例，为研究人员和从业者提供统一的参考，以保护具有数字双胞胎的硬件。本文还探讨了大型语言模型与数字双胞胎的集成以增强安全性，并讨论了当前的挑战、解决方案和未来的研究方向。



## **6. Steering MoE LLMs via Expert (De)Activation**

通过专家（去）激活MoE LLM cs.CL

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09660v1) [paper-pdf](http://arxiv.org/pdf/2509.09660v1)

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.

摘要: 大型语言模型（LLM）中的专家混合（MoE）通过专门的前向网络（FFN）子集（称为专家）路由每个令牌。我们介绍了SteerMoE，这是一个通过检测和控制与行为相关的专家来引导MoE模型的框架。我们的检测方法识别出在表现出相反行为的成对输入中具有不同激活模式的专家。通过在推理过程中选择性地（去）激活此类专家，我们可以在无需重新培训或修改权重的情况下控制忠诚和安全等行为。在11个基准和6个LLM中，我们的指导将安全性提高了+20%，忠诚度提高了+27%。在对抗性攻击模式下，它单独会降低-41%，与现有越狱方法结合使用时，安全性会降低-100%，绕过所有安全护栏，暴露了隐藏在专家体内的对齐伪造的新维度。



## **7. Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks**

LLM可以黑客攻击企业网络吗？自主假设漏洞渗透测试Active目录网络 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2502.04227v3) [paper-pdf](http://arxiv.org/pdf/2502.04227v3)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Enterprise penetration-testing is often limited by high operational costs and the scarcity of human expertise. This paper investigates the feasibility and effectiveness of using Large Language Model (LLM)-driven autonomous systems to address these challenges in real-world Active Directory (AD) enterprise networks.   We introduce a novel prototype designed to employ LLMs to autonomously perform Assumed Breach penetration-testing against enterprise networks. Our system represents the first demonstration of a fully autonomous, LLM-driven framework capable of compromising accounts within a real-life Microsoft Active Directory testbed, GOAD.   We perform our empirical evaluation using five LLMs, comparing reasoning to non-reasoning models as well as including open-weight models. Through quantitative and qualitative analysis, incorporating insights from cybersecurity experts, we demonstrate that autonomous LLMs can effectively conduct Assumed Breach simulations. Key findings highlight their ability to dynamically adapt attack strategies, perform inter-context attacks (e.g., web-app audits, social engineering, and unstructured data analysis for credentials), and generate scenario-specific attack parameters like realistic password candidates. The prototype exhibits robust self-correction mechanisms, installing missing tools and rectifying invalid command generations.   We find that the associated costs are competitive with, and often significantly lower than, those incurred by professional human pen-testers, suggesting a path toward democratizing access to essential security testing for organizations with budgetary constraints. However, our research also illuminates existing limitations, including instances of LLM ``going down rabbit holes'', challenges in comprehensive information transfer between planning and execution modules, and critical safety concerns that necessitate human oversight.

摘要: 企业渗透测试通常受到高运营成本和人力专业知识稀缺的限制。本文研究了使用大型语言模型（LLM）驱动的自治系统来应对现实世界Active目录（AD）企业网络中这些挑战的可行性和有效性。   我们引入了一种新颖的原型，旨在使用LLM来针对企业网络自主执行假设突破渗透测试。我们的系统首次演示了完全自治的LLM驱动框架，该框架能够在现实生活中的Microsoft Active目录测试平台GOAD中危及帐户。   我们使用五个LLM进行实证评估，将推理与非推理模型进行比较，并包括开放权重模型。通过定量和定性分析，结合网络安全专家的见解，我们证明自治LLM可以有效地进行假设漏洞模拟。主要发现强调了它们动态调整攻击策略、执行上下文间攻击（例如，Web应用程序审计、社会工程和凭证的非结构化数据分析），并生成特定于集群的攻击参数，就像现实的候选密码一样。该原型展示了强大的自我纠正机制，可以安装缺失的工具并纠正无效的命令生成。   我们发现，相关成本与专业人类笔测试人员产生的成本具有竞争力，并且往往远低于专业人类笔测试人员的成本，这为预算限制的组织提供了一条民主化的途径。然而，我们的研究也揭示了现有的局限性，包括LLM“掉进兔子洞”的例子、规划和执行模块之间全面信息传输的挑战，以及需要人工监督的关键安全问题。



## **8. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

IDEATOR：使用自己越狱和基准大型视觉语言模型 cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.00827v5) [paper-pdf](http://arxiv.org/pdf/2411.00827v5)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.VLJailbreakBench is publicly available at https://roywang021.github.io/VLJailbreakBench.

摘要: 随着大型视觉语言模型（VLM）的日益突出，确保其安全部署变得至关重要。最近的研究探索了VLM针对越狱攻击的鲁棒性--利用模型漏洞来引发有害输出的技术。然而，多样化多模式数据的可用性有限，限制了当前的方法严重依赖于从有害文本数据集派生的对抗性或手动制作的图像，而这些图像通常缺乏跨不同背景的有效性和多样性。本文中，我们提出了IDEATOR，这是一种新型越狱方法，可以自主生成用于黑匣子越狱攻击的恶意图像-文本对。IDEATOR基于这样的见解：VLM本身可以充当强大的红队模型，用于生成多模式越狱提示。具体来说，IDEATOR利用VLM创建有针对性的越狱文本，并将其与由最先进的扩散模型生成的越狱图像配对。大量实验证明了IDEATOR的高效率和可移植性，在越狱MiniGPT-4中平均只需5.34次查询即可实现94%的攻击成功率（ASB），转移到LLaVA、INSTBLIP和Chameleon时，攻击成功率分别为82%、88%和75%。基于IDEATOR强大的可移植性和自动化流程，我们推出了VLJailbreakBench，这是一个由3，654个多模式越狱样本组成的安全基准。我们对最近发布的11个VLM的基准结果揭示了安全一致方面的显着差距。例如，我们的挑战集在GPT-4 o上实现了46.31%的ASB，在Claude-3.5-Sonnet上实现了19.65%的ASB，这凸显了对更强防御的迫切需要。VLJailbreakBench可在https://roywang021.github.io/VLJailbreakBench上公开获取。



## **9. SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models**

SimMark：一种针对大型语言模型的稳健基于句子级相似性的水印算法 cs.CL

Accepted to EMNLP 25 main

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2502.02787v2) [paper-pdf](http://arxiv.org/pdf/2502.02787v2)

**Authors**: Amirhossein Dabiriaghdam, Lele Wang

**Abstract**: The widespread adoption of large language models (LLMs) necessitates reliable methods to detect LLM-generated text. We introduce SimMark, a robust sentence-level watermarking algorithm that makes LLMs' outputs traceable without requiring access to model internals, making it compatible with both open and API-based LLMs. By leveraging the similarity of semantic sentence embeddings combined with rejection sampling to embed detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while maintaining the text quality and fluency.

摘要: 大型语言模型（LLM）的广泛采用需要可靠的方法来检测LLM生成的文本。我们引入了SimMark，这是一种强大的业务级水印算法，可以使LLM的输出可追溯，而无需访问模型内部，使其与开放式和基于API的LLM兼容。通过利用语义句子嵌入的相似性与拒绝抽样相结合来嵌入人类无法感知的可检测统计模式，并采用软计数机制，SimMark实现了针对重述攻击的鲁棒性。实验结果表明，SimMark为LLM生成的内容的鲁棒性水印设定了新的基准，在鲁棒性、采样效率和跨不同领域的适用性方面超越了先前的业务级水印技术，同时保持文本质量和流畅性。



## **10. Character-Level Perturbations Disrupt LLM Watermarks**

初级扰动扰乱LLM水印 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09112v1) [paper-pdf](http://arxiv.org/pdf/2509.09112v1)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.

摘要: 大型语言模型（LLM）水印将可检测信号嵌入到生成的文本中，以实现版权保护、防止滥用和内容检测。虽然之前的研究使用水印去除攻击来评估稳健性，但这些方法通常不是最优的，从而产生了一种误解，即有效的去除需要大的扰动或强大的对手。   为了弥合差距，我们首先形式化LLM水印的系统模型，并描述了两个受有限访问水印检测器限制的现实威胁模型。然后，我们分析不同类型的扰动在其攻击范围内的变化，即，一次编辑可以影响的代币数量。我们观察到字符级扰动（例如，拼写错误、互换、删除、同字形）可以通过扰乱标记化过程同时影响多个标记。我们证明，在最严格的威胁模型下，字符级扰动对于水印去除来说明显更有效。我们进一步提出了基于遗传算法（GA）的引导删除攻击，使用一个参考检测器进行优化。在一个实际的威胁模型与有限的黑盒查询的水印检测器，我们的方法表现出强大的去除性能。实验证实了字符级扰动的优越性和遗传算法在现实约束条件下去除水印的有效性。此外，我们认为有一个对抗性的困境时，考虑潜在的防御：任何固定的防御可以绕过一个合适的扰动策略。受此原则的启发，我们提出了一种自适应复合字符级攻击。实验结果表明，这种方法可以有效地击败防御。我们的研究结果强调了现有LLM水印方案中的重大漏洞，并强调了开发新的稳健机制的紧迫性。



## **11. ACE: A Security Architecture for LLM-Integrated App Systems**

ACE：LLM集成应用程序系统的安全架构 cs.CR

25 pages, 13 figures, 8 tables; accepted by Network and Distributed  System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2504.20984v3) [paper-pdf](http://arxiv.org/pdf/2504.20984v3)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that ACE is secure against attacks from the InjecAgent and Agent Security Bench benchmarks for indirect prompt injection, and our newly introduced attacks. We also evaluate the utility of ACE in realistic environments, using the Tool Usage suite from the LangChain benchmark. Our architecture represents a significant advancement towards hardening LLM-based systems using system security principles.

摘要: LLM集成的应用程序系统通过第三方应用程序扩展了大型语言模型（LLM）的实用性，第三方应用程序由系统LLM使用交错的规划和执行阶段调用，以回答用户查询。这些系统引入了新的攻击载体，恶意应用程序可能会导致规划或执行的完整性违反、可用性崩溃或执行期间的隐私受到损害。   在这项工作中，我们识别了影响规划完整性以及LLM集成应用程序中执行完整性和可用性的新攻击，并针对IsolateGPT（旨在减轻恶意应用程序攻击的最新解决方案）进行演示。我们提出Abstract-Concrete-Execute（ACE），这是一种针对LLM集成应用程序系统的新安全架构，为系统规划和执行提供安全保障。具体来说，ACE将规划分为两个阶段，首先仅使用可信信息创建抽象执行计划，然后使用已安装的系统应用程序将抽象计划映射到具体计划。我们通过对结构化计划输出的静态分析来验证系统生成的计划是否满足用户指定的安全信息流约束。在执行过程中，ACE在应用程序之间强制设置数据和能力障碍，并确保执行按照可信的抽象计划进行。我们通过实验表明，ACE可以抵御来自InjectAgent和Agent Security Bench间接即时注入基准的攻击以及我们新引入的攻击。我们还评估了ACE在现实环境中的实用性，使用LangChain基准测试中的工具使用套件。我们的架构代表了使用系统安全原则加强基于LLM的系统的重大进步。



## **12. Architecting Resilient LLM Agents: A Guide to Secure Plan-then-Execute Implementations**

构建弹性LLM代理：安全计划后执行实施指南 cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08646v1) [paper-pdf](http://arxiv.org/pdf/2509.08646v1)

**Authors**: Ron F. Del Rosario, Klaudia Krawiecka, Christian Schroeder de Witt

**Abstract**: As Large Language Model (LLM) agents become increasingly capable of automating complex, multi-step tasks, the need for robust, secure, and predictable architectural patterns is paramount. This paper provides a comprehensive guide to the ``Plan-then-Execute'' (P-t-E) pattern, an agentic design that separates strategic planning from tactical execution. We explore the foundational principles of P-t-E, detailing its core components - the Planner and the Executor - and its architectural advantages in predictability, cost-efficiency, and reasoning quality over reactive patterns like ReAct (Reason + Act). A central focus is placed on the security implications of this design, particularly its inherent resilience to indirect prompt injection attacks by establishing control-flow integrity. We argue that while P-t-E provides a strong foundation, a defense-in-depth strategy is necessary, and we detail essential complementary controls such as the Principle of Least Privilege, task-scoped tool access, and sandboxed code execution. To make these principles actionable, this guide provides detailed implementation blueprints and working code references for three leading agentic frameworks: LangChain (via LangGraph), CrewAI, and AutoGen. Each framework's approach to implementing the P-t-E pattern is analyzed, highlighting unique features like LangGraph's stateful graphs for re-planning, CrewAI's declarative tool scoping for security, and AutoGen's built-in Docker sandboxing. Finally, we discuss advanced patterns, including dynamic re-planning loops, parallel execution with Directed Acyclic Graphs (DAGs), and the critical role of Human-in-the-Loop (HITL) verification, to offer a complete strategic blueprint for architects, developers, and security engineers aiming to build production-grade, resilient, and trustworthy LLM agents.

摘要: 随着大型语言模型（LLM）代理越来越有能力自动化复杂的多步骤任务，对稳健、安全且可预测的架构模式的需求变得至关重要。本文提供了“计划然后执行”（P-t-E）模式的全面指南，这是一种将战略规划与战术执行分开的代理设计。我们探索了P-t-E的基本原则，详细介绍了其核心组件--计划者和执行者--以及与ReAct（Reason + Act）等反应式模式相比，其在可预测性、成本效率和推理质量方面的架构优势。中心焦点是该设计的安全影响，特别是其通过建立控制流完整性来对间接即时注入攻击的固有弹性。我们认为，虽然P-t-E提供了坚实的基础，但深度防御策略是必要的，并且我们详细介绍了必要的补充控制，例如最小特权原则、任务范围的工具访问和沙箱代码执行。为了使这些原则具有可操作性，本指南为三个领先的代理框架提供了详细的实现蓝图和工作代码参考：LangChain（通过LangCurve）、CrewAI和AutoGen。分析了每个框架实现P-t-E模式的方法，重点介绍了Langstra用于重新规划的有状态图、CrewAI用于安全的声明性工具范围以及AutoGen的内置Docker沙箱等独特功能。最后，我们讨论了高级模式，包括动态重新规划循环、与有向无环图（DAB）的并行执行以及人在环（HITL）验证的关键作用，为旨在构建生产级、弹性且值得信赖的LLM代理的架构师、开发人员和安全工程师提供完整的战略蓝图。



## **13. Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors**

您的语言模型可以像人类一样秘密写作：对LLM生成的文本检测器的对比重述攻击 cs.CL

Accepted by EMNLP-2025

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2505.15337v3) [paper-pdf](http://arxiv.org/pdf/2505.15337v3)

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.

摘要: 学术抄袭等大型语言模型（LLM）的滥用推动了识别LLM生成文本的检测器的发展。为了绕过这些检测器，出现了故意重写这些文本以逃避检测的重述攻击。尽管取得了成功，但现有方法需要大量的数据和计算预算来训练专门的解释器，并且当面对先进的检测算法时，它们的攻击功效会大大降低。为了解决这个问题，我们提出了\textBF{Co} ntrasive\textBF{P}araphrase \textBF{A}ttack（CoPA），这是一种免训练方法，可以使用现成的LLM有效地欺骗文本检测器。第一步是仔细编写指令，鼓励LLM生成更多类似人类的文本。尽管如此，我们观察到LLM固有的统计偏差仍然会导致一些生成的文本携带某些可以被检测器捕获的类似机器的属性。为了克服这个问题，CoPA构建了一个辅助的类似机器的单词分布，与LLM生成的类似人类的分布形成对比。通过在解码过程中从类人分布中减去类机器模式，CoPA能够生成文本检测器难以识别的句子。我们的理论分析表明了拟议攻击的优越性。大量实验验证了CoPA在各种场景中欺骗文本检测器的有效性。



## **14. TraceRAG: A LLM-Based Framework for Explainable Android Malware Detection and Behavior Analysis**

TraceRAG：一个基于LLM的可解释Android恶意软件检测和行为分析框架 cs.SE

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08865v1) [paper-pdf](http://arxiv.org/pdf/2509.08865v1)

**Authors**: Guangyu Zhang, Xixuan Wang, Shiyu Sun, Peiyan Xiao, Kun Sun, Yanhai Xiong

**Abstract**: Sophisticated evasion tactics in malicious Android applications, combined with their intricate behavioral semantics, enable attackers to conceal malicious logic within legitimate functions, underscoring the critical need for robust and in-depth analysis frameworks. However, traditional analysis techniques often fail to recover deeply hidden behaviors or provide human-readable justifications for their decisions. Inspired by advances in large language models (LLMs), we introduce TraceRAG, a retrieval-augmented generation (RAG) framework that bridges natural language queries and Java code to deliver explainable malware detection and analysis. First, TraceRAG generates summaries of method-level code snippets, which are indexed in a vector database. At query time, behavior-focused questions retrieve the most semantically relevant snippets for deeper inspection. Finally, based on the multi-turn analysis results, TraceRAG produces human-readable reports that present the identified malicious behaviors and their corresponding code implementations. Experimental results demonstrate that our method achieves 96\% malware detection accuracy and 83.81\% behavior identification accuracy based on updated VirusTotal (VT) scans and manual verification. Furthermore, expert evaluation confirms the practical utility of the reports generated by TraceRAG.

摘要: 恶意Android应用程序中复杂的规避策略，再加上复杂的行为语义，使攻击者能够将恶意逻辑隐藏在合法功能中，凸显了对稳健和深入分析框架的迫切需求。然而，传统的分析技术往往无法恢复隐藏的行为，也无法为其决策提供人类可读的理由。受大型语言模型（LLM）进步的启发，我们引入了TraceRAG，这是一种检索增强生成（RAG）框架，它将自然语言查询和Java代码连接起来，以提供可解释的恶意软件检测和分析。首先，TraceRAG生成方法级代码片段的摘要，这些代码片段在载体数据库中编入索引。在查询时，以行为为中心的问题检索最相关的语义片段以进行更深入的检查。最后，根据多轮分析结果，TraceRAG生成人类可读的报告，其中呈现识别的恶意行为及其相应的代码实现。实验结果表明，基于更新的Virus Total（VT）扫描和手动验证，我们的方法实现了96%的恶意软件检测准确率和83.81%的行为识别准确率。此外，专家评估证实了TraceRAG生成的报告的实际用途。



## **15. ImportSnare: Directed "Code Manual" Hijacking in Retrieval-Augmented Code Generation**

ImportSnare：检索增强代码生成中的定向“代码手册”劫持 cs.CR

This paper has been accepted by the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07941v1) [paper-pdf](http://arxiv.org/pdf/2509.07941v1)

**Authors**: Kai Ye, Liangcai Su, Chenxiong Qian

**Abstract**: Code generation has emerged as a pivotal capability of Large Language Models(LLMs), revolutionizing development efficiency for programmers of all skill levels. However, the complexity of data structures and algorithmic logic often results in functional deficiencies and security vulnerabilities in generated code, reducing it to a prototype requiring extensive manual debugging. While Retrieval-Augmented Generation (RAG) can enhance correctness and security by leveraging external code manuals, it simultaneously introduces new attack surfaces.   In this paper, we pioneer the exploration of attack surfaces in Retrieval-Augmented Code Generation (RACG), focusing on malicious dependency hijacking. We demonstrate how poisoned documentation containing hidden malicious dependencies (e.g., matplotlib_safe) can subvert RACG, exploiting dual trust chains: LLM reliance on RAG and developers' blind trust in LLM suggestions. To construct poisoned documents, we propose ImportSnare, a novel attack framework employing two synergistic strategies: 1)Position-aware beam search optimizes hidden ranking sequences to elevate poisoned documents in retrieval results, and 2)Multilingual inductive suggestions generate jailbreaking sequences to manipulate LLMs into recommending malicious dependencies. Through extensive experiments across Python, Rust, and JavaScript, ImportSnare achieves significant attack success rates (over 50% for popular libraries such as matplotlib and seaborn) in general, and is also able to succeed even when the poisoning ratio is as low as 0.01%, targeting both custom and real-world malicious packages. Our findings reveal critical supply chain risks in LLM-powered development, highlighting inadequate security alignment for code generation tasks. To support future research, we will release the multilingual benchmark suite and datasets. The project homepage is https://importsnare.github.io.

摘要: 代码生成已成为大型语言模型（LLM）的关键功能，彻底改变了所有技能水平的程序员的开发效率。然而，数据结构和算法逻辑的复杂性往往会导致生成的代码中存在功能缺陷和安全漏洞，从而将其简化为需要大量手动调试的原型。虽然检索增强生成（RAG）可以通过利用外部代码手册来增强正确性和安全性，但它同时引入了新的攻击表面。   在本文中，我们率先探索检索增强代码生成（RACG）中的攻击表面，重点关注恶意依赖劫持。我们演示了如何包含隐藏恶意依赖项的有毒文档（例如，matplotlib_safe）可以利用双重信任链颠覆RACG：LLM对RAG的依赖以及开发人员对LLM建议的盲目信任。为了构建有毒文档，我们提出了ImportSnare，这是一种采用两种协同策略的新型攻击框架：1）位置感知射束搜索优化隐藏排名序列，以提升检索结果中的有毒文档，2）多语言归纳建议生成越狱序列，以操纵LLM推荐恶意依赖项。通过Python、Rust和JavaScript的广泛实验，ImportSnare总体上实现了显着的攻击成功率（matplotlib和seaborn等流行库超过50%），并且即使中毒率低至0.01%，也能够成功，针对自定义和现实世界的恶意软件包。我们的研究结果揭示了LLM驱动的开发中的关键供应链风险，凸显了代码生成任务的安全一致性不足。为了支持未来的研究，我们将发布多语言基准套件和数据集。项目主页为https://importsnare.github.io。



## **16. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

使用结构化攻击树的LLM驱动渗透测试中的引导推理 cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07939v1) [paper-pdf](http://arxiv.org/pdf/2509.07939v1)

**Authors**: Katsuaki Nakano, Reza Feyyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments

摘要: 大型语言模型（LLM）的最新进展激发了人们对自动化网络安全渗透测试工作流程的兴趣，为企业系统提供更快、更一致的漏洞评估。现有的用于渗透测试的LLM代理主要依赖于自我引导推理，这可能会产生不准确或幻觉的程序步骤。因此，LLM代理可能会采取非生产性的行动，例如利用未使用的软件库或生成重复先前策略的周期性响应。在这项工作中，我们提出了一个用于渗透测试LLM代理的引导推理管道，该管道结合了从MITRE ATT & CK矩阵（一个经过验证的渗透测试kll链）构建的确定性任务树，以将LLM的重组过程限制为明确定义的策略、技术和程序。这将推理锚定在经过验证的渗透测试方法中，并通过引导代理走向更有成效的攻击程序来过滤无效操作。为了评估我们的方法，我们使用三个LLM（Llama-3-8B、Gemini-1.5和GPT-4）构建了一个自动渗透测试LLM代理，并将其应用于导航10个HackTheBox网络安全演习，其中包含103个代表现实世界网络攻击场景的离散子任务。我们提出的推理管道使用Llama-3-8B、Gemini-1.5和GPT-4分别引导LLM代理完成71.8%、72.8%和78.6%的子任务。相比之下，使用自我引导推理的最先进的LLM渗透测试工具仅完成了13.5%、16.5%和75.7%的子任务，并且需要86.2%、118.7%和205.9%的模型查询。这表明将确定性任务树纳入LLM推理管道可以提高自动化网络安全评估的准确性和效率



## **17. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

大型多模式模型的鲁棒适应用于检索增强仇恨模因检测 cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2502.13061v3) [paper-pdf](http://arxiv.org/pdf/2502.13061v3)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

摘要: 仇恨模因已成为互联网上的一个重要问题，需要强大的自动化检测系统。虽然大型多模式模型（LSYS）在仇恨模因检测方面表现出了希望，但它们面临着显着的挑战，例如次优的性能和有限的域外概括能力。最近的研究进一步揭示了在这种环境下将监督微调（SFT）和上下文学习应用于LSYS时的局限性。为了解决这些问题，我们提出了一个用于仇恨模因检测的鲁棒适应框架，该框架可以增强领域内准确性和跨领域概括性，同时保留Letts的一般视觉语言能力。分析表明，与SFT模型相比，我们的方法在对抗攻击下实现了更好的鲁棒性。对六个模因分类数据集的实验表明，我们的方法实现了最先进的性能，优于更大的代理系统。此外，与标准SFT相比，我们的方法为解释仇恨内容生成了更高质量的理由，增强了模型的可解释性。代码可访问https://github.com/JingbiaoMei/RGCL



## **18. AgentSentinel: An End-to-End and Real-Time Security Defense Framework for Computer-Use Agents**

AgentSentinel：计算机使用代理的端到端实时安全防御框架 cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07764v1) [paper-pdf](http://arxiv.org/pdf/2509.07764v1)

**Authors**: Haitao Hu, Peng Chen, Yanpeng Zhao, Yuqi Chen

**Abstract**: Large Language Models (LLMs) have been increasingly integrated into computer-use agents, which can autonomously operate tools on a user's computer to accomplish complex tasks. However, due to the inherently unstable and unpredictable nature of LLM outputs, they may issue unintended tool commands or incorrect inputs, leading to potentially harmful operations. Unlike traditional security risks stemming from insecure user prompts, tool execution results from LLM-driven decisions introduce new and unique security challenges. These vulnerabilities span across all components of a computer-use agent. To mitigate these risks, we propose AgentSentinel, an end-to-end, real-time defense framework designed to mitigate potential security threats on a user's computer. AgentSentinel intercepts all sensitive operations within agent-related services and halts execution until a comprehensive security audit is completed. Our security auditing mechanism introduces a novel inspection process that correlates the current task context with system traces generated during task execution. To thoroughly evaluate AgentSentinel, we present BadComputerUse, a benchmark consisting of 60 diverse attack scenarios across six attack categories. The benchmark demonstrates a 87% average attack success rate on four state-of-the-art LLMs. Our evaluation shows that AgentSentinel achieves an average defense success rate of 79.6%, significantly outperforming all baseline defenses.

摘要: 大型语言模型（LLM）已越来越多地集成到计算机使用代理中，这些代理可以在用户计算机上自主操作工具来完成复杂的任务。然而，由于LLM输出本质上不稳定和不可预测，它们可能会发出意外的工具命令或不正确的输入，从而导致潜在的有害操作。与由不安全的用户提示产生的传统安全风险不同，LLM驱动的决策产生的工具执行会带来新的独特安全挑战。这些漏洞跨越计算机使用代理的所有组件。为了减轻这些风险，我们提出了AgentSentinel，这是一种端到端的实时防御框架，旨在减轻用户计算机上的潜在安全威胁。AgentSentinel拦截代理相关服务中的所有敏感操作，并停止执行，直到完成全面的安全审计。我们的安全审计机制引入了一种新颖的检查过程，该过程将当前任务上下文与任务执行期间生成的系统跟踪关联起来。为了彻底评估AgentSentinel，我们介绍了BadComputerUse，这是一个由六种攻击类别的60种不同攻击场景组成的基准。该基准显示，四种最先进的LLM的平均攻击成功率为87%。我们的评估显示，AgentSentinel的平均防御成功率为79.6%，显着优于所有基线防御。



## **19. TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation**

TrojanRobot：针对基于VLM的机器人操纵的物理世界后门攻击 cs.RO

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2411.11683v4) [paper-pdf](http://arxiv.org/pdf/2411.11683v4)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Aishan Liu, Leo Yu Zhang, Xiaohua Jia

**Abstract**: Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.

摘要: \textit{大型语言模型}（LLM）和\textit{视觉语言模型}（VLMS）利用它们的理解和感知能力，越来越多地增强物理世界中的机器人操纵能力。最近，针对此类机器人策略的各种攻击被提出，其中后门攻击因其高隐身性和强持久性能力而引起了相当大的关注。然而，现有的后门工作仅限于模拟器，并且受到物理世界实现的影响。为了解决这个问题，我们提出了\textit{TrojanRobot}，这是物理世界中一种高度隐蔽且广泛有效的机器人后门攻击。具体来说，我们通过将后门模块嵌入模块到模块化机器人策略中来引入模块中毒方法，从而对策略的视觉感知模块进行后门控制，从而后门化整个机器人策略。我们的普通实现利用一个经过后门微调的VLM作为后门模块。为了增强其在物理环境中的通用性，我们提出了一种主要实现，利用LVLM作为后门范式并开发三种类型的主要攻击，即\textit{perspective}、\textit{staduction}和\textit{intentional}攻击，从而实现更细粒度的后门。UR 3e机械手与18个任务指令使用机器人策略的基础上，四个VLMs的广泛实验证明了广泛的有效性和物理世界的隐身TrojanRobot。我们的攻击视频演示可以通过github链接https://trojanrobot.github.io获得。



## **20. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

通过激活引导MCMC采样的可转移直接即时注射 cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.

摘要: 直接提示注入（DPI）攻击由于其低执行门槛和高潜在危害性，对大型语言模型（LLM）构成了严重的安全威胁。针对现有白盒/灰盒方法的不实用性和黑盒方法的可移植性差的问题，提出了一种激活引导的提示注入攻击框架.首先，我们构建了一个基于能量的模型（EBM）使用代理模型的激活来评估对抗性提示的质量。在经过训练的EBM的指导下，我们采用令牌级马尔可夫链蒙特卡罗（MCMC）采样来自适应地优化对抗性提示，从而实现无梯度的黑盒攻击。实验结果证明了我们卓越的跨模型可移植性，在五种主流LLM中实现了49.6%的攻击成功率（ASB），比人工制作的提示提高了34.6%，并在未见的任务场景中保持了36.6%的ASB。可解释性分析揭示了激活和攻击有效性之间的相关性，凸显了语义模式在可转移漏洞利用中的关键作用。



## **21. A Decade-long Landscape of Advanced Persistent Threats: Longitudinal Analysis and Global Trends**

长达十年的高级持续威胁格局：纵向分析和全球趋势 cs.CR

18 pages, 13 figures (including subfigures), 11 tables. To appear in  the Proceedings of the ACM Conference on Computer and Communications Security  (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07457v1) [paper-pdf](http://arxiv.org/pdf/2509.07457v1)

**Authors**: Shakhzod Yuldoshkhujaev, Mijin Jeon, Doowon Kim, Nick Nikiforakis, Hyungjoon Koo

**Abstract**: An advanced persistent threat (APT) refers to a covert, long-term cyberattack, typically conducted by state-sponsored actors, targeting critical sectors and often remaining undetected for long periods. In response, collective intelligence from around the globe collaborates to identify and trace surreptitious activities, generating substantial documentation on APT campaigns publicly available on the web. While prior works predominantly focus on specific aspects of APT cases, such as detection, evaluation, cyber threat intelligence, and dataset creation, limited attention has been devoted to revisiting and investigating these scattered dossiers in a longitudinal manner. The objective of our study is to fill the gap by offering a macro perspective, connecting key insights and global trends in past APT attacks. We systematically analyze six reliable sources-three focused on technical reports and another three on threat actors-examining 1,509 APT dossiers (24,215 pages) spanning 2014-2023, and identifying 603 unique APT groups worldwide. To efficiently unearth relevant information, we employ a hybrid methodology that combines rule-based information retrieval with large-language-model-based search techniques. Our longitudinal analysis reveals shifts in threat actor activities, global attack vectors, changes in targeted sectors, and relationships between cyberattacks and significant events such as elections or wars, which provide insights into historical patterns in APT evolution. Over the past decade, 154 countries have been affected, primarily using malicious documents and spear phishing as dominant initial infiltration vectors, with a noticeable decline in zero-day exploitation since 2016. Furthermore, we present our findings through interactive visualization tools, such as an APT map or flow diagram, to facilitate intuitive understanding of global patterns and trends in APT activities.

摘要: 高级持续威胁（APT）是指秘密的长期网络攻击，通常由国家支持的行为者实施，针对关键部门，并且通常长时间未被发现。作为回应，来自全球各地的集体情报机构合作识别和追踪秘密活动，在网上公开生成有关APT活动的大量文档。虽然之前的工作主要集中在APT案例的特定方面，例如检测、评估、网络威胁情报和数据集创建，但人们对以纵向方式重新访问和调查这些分散的档案的注意力有限。我们研究的目标是通过提供宏观视角、联系过去APT攻击的关键见解和全球趋势来填补这一空白。我们系统地分析了六个可靠的来源，其中三个集中在技术报告上，另外三个集中在威胁行为者上，检查了2014-2023年的1，509份APT档案（24，215页），并确定了全球603个独特的APT组织。为了有效地挖掘相关信息，我们采用了一种混合的方法，结合基于规则的信息检索与基于大语言模型的搜索技术。我们的纵向分析揭示了威胁行为者活动的变化、全球攻击载体、目标行业的变化以及网络攻击与选举或战争等重大事件之间的关系，从而深入了解APT演变的历史模式。在过去的十年中，154个国家受到影响，主要使用恶意文件和鱼叉式网络钓鱼作为主要的初始渗透载体，自2016年以来零日攻击明显下降。此外，我们还通过交互式可视化工具（例如APT地图或流程图）展示我们的发现，以促进对APT活动的全球模式和趋势的直观理解。



## **22. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models**

Obliviate：针对大型语言模型的稳健且实用的机器去学习 cs.CL

To appear at EMNLP 25 main conference

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.04416v2) [paper-pdf](http://arxiv.org/pdf/2505.04416v2)

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose \textbf{OBLIVIATE}, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA) ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: \emph{forget quality} (via a new document-level memorization score), \emph{model utility}, and \emph{fluency}. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios.

摘要: 在广泛的库中训练的大型语言模型（LLM）存在记忆敏感、受版权保护或有毒内容的风险。为了解决这个问题，我们提出了\textBF{ObLIVIATE}，这是一个强大的去学习框架，可以在保留模型效用的同时删除目标数据。该框架遵循一个结构化过程：提取目标令牌、构建保留集以及使用定制的损失函数进行微调，该函数包括三个部分--掩蔽、蒸馏和世界事实。使用低级适配器（LoRA）可以在不影响取消学习质量的情况下确保效率。我们使用一套全面的指标对多个数据集进行实验，包括《哈利·波特》系列、WMDP和TOFU：\{忘记质量}（通过新的文档级记忆评分）、\{模型实用程序}和\{流利度}。结果证明了它在抵抗隶属度推理攻击、最大限度地减少对保留数据的影响以及在不同场景下保持稳健性方面的有效性。



## **23. GRADA: Graph-based Reranking against Adversarial Documents Attack**

GRADA：基于图的重新排名对抗文档攻击 cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们的方法的有效性：GPT-3.5-Turbo，GPT-4 o，Llama3.1-8b，Llama3.1- 70 b和Qwen2.5- 7 b。我们使用三个数据集来评估性能，来自Natural Questions数据集的结果表明攻击成功率降低了80%，同时保持了最小的准确性损失。



## **24. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

多轮会话中的社会工程个性化攻击：LLM Agent仿真与检测 cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大型语言模型（LLM）驱动的聊天机器人，对社交媒体平台构成了社会工程（SE）攻击的重大风险。由于这些会话的动态性质，多回合、基于聊天的交互中的SE检测比单实例检测复杂得多。减轻这种威胁的一个关键因素是了解SE攻击的机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何影响他们的易感性。在这项工作中，我们提出了一个LLM-agentic框架，SE-VSim，模拟SE攻击机制，通过生成多轮对话。我们对具有不同性格特征的受害者特工进行建模，以评估心理特征如何影响操纵的易感性。我们使用包含1000多个模拟对话的数据集，检查了冒充招聘人员、资助机构和记者的对手试图提取敏感信息的攻击场景。基于此分析，我们提出了一个概念验证SE-OmniGuard，通过利用受害者个性的先验知识、评估攻击策略以及监控对话中的信息交换以识别潜在的SE尝试，为用户提供个性化保护。



## **25. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

6 pages, 2 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.17674v2) [paper-pdf](http://arxiv.org/pdf/2508.17674v2)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **26. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2501.01872v4) [paper-pdf](http://arxiv.org/pdf/2501.01872v4)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 尽管大型语言模型与人类价值观和道德原则广泛一致，但仍然容易受到利用其推理能力的复杂越狱攻击。现有的安全措施通常检测到明显的恶意意图，但无法解决微妙的、推理驱动的漏洞。在这项工作中，我们引入了POATE（极反相查询生成、对抗模板构建和搜索），这是一种新颖的越狱技术，利用对比推理来引发不道德的反应。POATE精心设计了语义上相反的意图，并将它们与对抗模板集成，以非凡的微妙性引导模型走向有害的输出。我们对参数大小不同的六个不同语言模型家族进行了广泛的评估，以证明攻击的稳健性，与现有方法相比，实现了显着更高的攻击成功率（~44%）。为了解决这个问题，我们提出了意图感知CoT和反向思维CoT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的稳健性并加强了模型对对抗性利用的防御。



## **27. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

注意您的服务器：对CP生态系统的寄生工具链攻击的系统研究 cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06572v1) [paper-pdf](http://arxiv.org/pdf/2509.06572v1)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.

摘要: 大型语言模型（LLM）通过模型上下文协议（HCP）越来越多地与外部系统集成，该协议使工具调用同步化，并已迅速成为LLM支持的应用程序的支柱。虽然这种范式增强了功能，但它也引入了根本性的安全转变：LLM从被动信息处理器过渡到面向任务的工具链的自主编排，扩大了攻击面，将对抗目标从操纵单个输出提升到劫持整个执行流。在本文中，我们揭示了一类新的攻击，即寄生工具链攻击，实例化为LCP无意隐私泄露（MCP-UPD）。这些攻击不需要受害者直接互动;相反，对手会将恶意指令嵌入到LLM在合法任务期间访问的外部数据源中。恶意逻辑渗透到工具链中，并分三个阶段展开：寄生摄入、隐私收集和隐私披露，最终导致私人数据的秘密泄露。我们的根本原因分析表明，LCP缺乏上下文工具隔离和最低特权强制执行，使得对抗指令能够不受限制地传播到敏感工具调用中。为了评估严重性，我们设计了MCP-SEC，并对LCP生态系统进行了首次大规模安全普查，分析了1，360台服务器上的12，230个工具。我们的研究结果表明，LCP生态系统中充斥着可利用的小工具和多样化的攻击方法，凸显了LCP平台的系统性风险以及LLM集成环境中对防御机制的迫切需求。



## **28. Soft Token Attacks Cannot Reliably Audit Unlearning in Large Language Models**

软令牌攻击无法可靠地审计大型语言模型中的取消学习 cs.CL

EMNLP 2025 Findings

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2502.15836v2) [paper-pdf](http://arxiv.org/pdf/2502.15836v2)

**Authors**: Haokun Chen, Sebastian Szyller, Weilin Xu, Nageen Himayat

**Abstract**: Large language models (LLMs) are trained using massive datasets, which often contain undesirable content such as harmful texts, personal information, and copyrighted material. To address this, machine unlearning aims to remove information from trained models. Recent work has shown that soft token attacks (STA) can successfully extract unlearned information from LLMs, but in this work we show that STAs can be an inadequate tool for auditing unlearning. Using common benchmarks such as Who Is Harry Potter? and TOFU, we demonstrate that in a strong auditor setting such attacks can elicit any information from the LLM, regardless of the deployed unlearning algorithm or whether the queried content was originally present in the training corpus. We further show that STA with just a few soft tokens (1-10) can elicit random strings over 400 characters long, indicating that STAs must be used carefully to effectively audit unlearning. Example code can be found at: https://github.com/IntelLabs/LLMart/tree/main/examples/unlearning

摘要: 大型语言模型（LLM）使用大量数据集进行训练，这些数据集通常包含有害文本，个人信息和版权材料等不良内容。为了解决这个问题，机器学习旨在从训练模型中删除信息。最近的工作表明，软令牌攻击（STA）可以成功地从LLM中提取未学习的信息，但在这项工作中，我们表明，STA可以是一个不充分的工具，审计unlearning。使用常见的基准，如谁是哈利波特？和TOFU，我们证明，在一个强大的审计设置这样的攻击可以引发任何信息的LLM，无论部署的unlearning算法或查询的内容是否最初存在于训练语料库。我们进一步表明，STA只有几个软令牌（1-10）可以引发超过400个字符长的随机字符串，这表明STA必须小心使用，以有效地审计遗忘。示例代码可以在https://github.com/IntelLabs/LLMart/tree/main/examples/unlearning上找到



## **29. Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks**

您的基于LLM的文本到SQL模型安全吗？通过后门攻击探索SQL注入 cs.CR

Accepted to SIGMOD 2026

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.05445v3) [paper-pdf](http://arxiv.org/pdf/2503.05445v3)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.

摘要: 大型语言模型（LLM）在将自然语言问题翻译为SQL查询（文本到SQL）方面显示出了最先进的结果，这是数据库界长期存在的挑战。然而，安全问题在很大程度上仍未得到探讨，特别是后门攻击的威胁，后门攻击可以通过对有毒数据集进行微调将恶意行为引入模型。在这项工作中，我们系统地研究了基于LLM的文本到SQL模型的漏洞，并提出了ToxicSQL，这是一种新型后门攻击框架。我们的方法利用隐形的（语义和字符级触发器）来使后门难以检测和删除，确保恶意行为保持隐蔽，同时对良性输入保持高模型准确性。此外，我们建议利用SQL注入有效负载作为后门目标，从而生成恶意但可执行的SQL查询，这在基于语言模型的SQL开发中构成了严重的安全和隐私风险。我们证明，仅注入0.44%的有毒数据就会导致79.41%的攻击成功率，对数据库安全构成重大风险。此外，我们还提出了检测和缓解策略来增强模型的可靠性。我们的研究结果强调了对安全意识的文本到SQL开发的迫切需求，并强调了针对后门威胁的强大防御的重要性。



## **30. Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?**

Mask-GCG：敌对后缀中的所有代币都是越狱攻击所必需的吗？ cs.CL

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06350v1) [paper-pdf](http://arxiv.org/pdf/2509.06350v1)

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks.

摘要: 对大型语言模型（LLM）的越狱攻击已经展示了各种成功的方法，攻击者通过这些方法操纵模型来生成他们旨在避免的有害响应。其中，贪婪坐标梯度（GCG）已成为一种通用且有效的方法，可以优化后缀中的标记以生成可越狱的提示。虽然已经提出了GCG的几种改进变体，但它们都依赖于固定长度的后缀。然而，这些后缀中的潜在冗余仍有待探索。在这项工作中，我们提出了Mask-GCG，这是一种即插即用方法，它采用可学习的标记掩蔽来识别后缀内有影响力的标记。我们的方法增加了高影响力位置上代币的更新概率，同时修剪低影响力位置上的代币。与GCG相比，这种修剪不仅减少了冗余，还减少了梯度空间的大小，从而降低了计算负担并缩短了实现成功攻击所需的时间。我们通过将Mask-GCG应用于原始GCG和几种改进的变体来评估Mask-GCG。实验结果表明，后缀中的大多数令牌对攻击成功做出了显着贡献，并且修剪少数低影响令牌不会影响损失值或损害攻击成功率（ASB），从而揭示了LLM提示中的令牌冗余。我们的研究结果为从越狱攻击的角度开发高效且可解释的LLM提供了见解。



## **31. Embedding Poisoning: Bypassing Safety Alignment via Embedding Semantic Shift**

嵌入中毒：通过嵌入语义转变来实现安全一致 cs.CR

16 pages,9 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06338v1) [paper-pdf](http://arxiv.org/pdf/2509.06338v1)

**Authors**: Shuai Yuan, Zhibo Zhang, Yuxi Li, Guangdong Bai, Wang Kailong

**Abstract**: The widespread distribution of Large Language Models (LLMs) through public platforms like Hugging Face introduces significant security challenges. While these platforms perform basic security scans, they often fail to detect subtle manipulations within the embedding layer. This work identifies a novel class of deployment phase attacks that exploit this vulnerability by injecting imperceptible perturbations directly into the embedding layer outputs without modifying model weights or input text. These perturbations, though statistically benign, systematically bypass safety alignment mechanisms and induce harmful behaviors during inference. We propose Search based Embedding Poisoning(SEP), a practical, model agnostic framework that introduces carefully optimized perturbations into embeddings associated with high risk tokens. SEP leverages a predictable linear transition in model responses, from refusal to harmful output to semantic deviation to identify a narrow perturbation window that evades alignment safeguards. Evaluated across six aligned LLMs, SEP achieves an average attack success rate of 96.43% while preserving benign task performance and evading conventional detection mechanisms. Our findings reveal a critical oversight in deployment security and emphasize the urgent need for embedding level integrity checks in future LLM defense strategies.

摘要: 通过Hugging Face等公共平台广泛分发大型语言模型（LLM）带来了重大的安全挑战。虽然这些平台执行基本的安全扫描，但它们通常无法检测到嵌入层内的微妙操纵。这项工作识别了一类新型的部署阶段攻击，这些攻击通过在不修改模型权重或输入文本的情况下直接将不可感知的扰动注入到嵌入层输出中来利用该漏洞。这些扰动虽然在统计上是良性的，但系统性地绕过了安全对齐机制，并在推理过程中引发有害行为。我们提出了基于搜索的嵌入中毒（SDP），这是一个实用的、模型不可知的框架，它将精心优化的扰动引入到与高风险代币相关的嵌入中。BEP利用模型响应中的可预测线性转变（从拒绝到有害输出再到语义偏差）来识别逃避对齐保障措施的狭窄扰动窗口。通过对六个对齐的LLM进行评估，SDP的平均攻击成功率为96.43%，同时保持良性任务性能并规避传统检测机制。我们的调查结果揭示了部署安全方面的严重疏忽，并强调迫切需要在未来的LLM防御策略中嵌入级别完整性检查。



## **32. AttestLLM: Efficient Attestation Framework for Billion-scale On-device LLMs**

AttestLLM：用于数十亿规模设备上LLM的高效认证框架 cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06326v1) [paper-pdf](http://arxiv.org/pdf/2509.06326v1)

**Authors**: Ruisi Zhang, Yifei Zhao, Neusha Javidnia, Mengxin Zheng, Farinaz Koushanfar

**Abstract**: As on-device LLMs(e.g., Apple on-device Intelligence) are widely adopted to reduce network dependency, improve privacy, and enhance responsiveness, verifying the legitimacy of models running on local devices becomes critical. Existing attestation techniques are not suitable for billion-parameter Large Language Models (LLMs), struggling to remain both time- and memory-efficient while addressing emerging threats in the LLM era. In this paper, we present AttestLLM, the first-of-its-kind attestation framework to protect the hardware-level intellectual property (IP) of device vendors by ensuring that only authorized LLMs can execute on target platforms. AttestLLM leverages an algorithm/software/hardware co-design approach to embed robust watermarking signatures onto the activation distributions of LLM building blocks. It also optimizes the attestation protocol within the Trusted Execution Environment (TEE), providing efficient verification without compromising inference throughput. Extensive proof-of-concept evaluations on LLMs from Llama, Qwen, and Phi families for on-device use cases demonstrate AttestLLM's attestation reliability, fidelity, and efficiency. Furthermore, AttestLLM enforces model legitimacy and exhibits resilience against model replacement and forgery attacks.

摘要: 作为设备上LLM（例如，Apple设备上智能）被广泛采用以减少网络依赖性、改善隐私和增强响应能力，验证在本地设备上运行的型号的合法性变得至关重要。现有的证明技术不适合数十亿参数的大型语言模型（LLM），在解决LLM时代新出现的威胁时，难以保持时间和内存效率。在本文中，我们介绍了AttestLLM，这是首个同类认证框架，旨在通过确保只有授权的LLM才能在目标平台上执行来保护设备供应商的硬件级知识产权（IP）。AttestLLM利用算法/软件/硬件协同设计方法将稳健的水印签名嵌入到LLM构建块的激活分布中。它还优化了可信执行环境（TEK）中的证明协议，在不影响推理吞吐量的情况下提供高效验证。针对设备上用例对Llama、Qwen和Phi系列的LLM进行了广泛的概念验证评估，证明了AttestLLM的证明可靠性、保真度和效率。此外，AttestLLM强制执行模型合法性，并表现出针对模型替换和伪造攻击的弹性。



## **33. Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs**

多模式即时注入攻击：现代LLM的风险和防御 cs.CR

8 pages, 4 figures, 2 tables

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2509.05883v1) [paper-pdf](http://arxiv.org/pdf/2509.05883v1)

**Authors**: Andrew Yeo, Daeseon Choi

**Abstract**: Large Language Models (LLMs) have seen rapid adoption in recent years, with industries increasingly relying on them to maintain a competitive advantage. These models excel at interpreting user instructions and generating human-like responses, leading to their integration across diverse domains, including consulting and information retrieval. However, their widespread deployment also introduces substantial security risks, most notably in the form of prompt injection and jailbreak attacks.   To systematically evaluate LLM vulnerabilities -- particularly to external prompt injection -- we conducted a series of experiments on eight commercial models. Each model was tested without supplementary sanitization, relying solely on its built-in safeguards. The results exposed exploitable weaknesses and emphasized the need for stronger security measures. Four categories of attacks were examined: direct injection, indirect (external) injection, image-based injection, and prompt leakage. Comparative analysis indicated that Claude 3 demonstrated relatively greater robustness; nevertheless, empirical findings confirm that additional defenses, such as input normalization, remain necessary to achieve reliable protection.

摘要: 近年来，大型语言模型（LLM）得到了迅速采用，行业越来越依赖它们来保持竞争优势。这些模型擅长解释用户指令并生成类似人类的响应，从而使它们能够在咨询和信息检索等不同领域进行集成。然而，它们的广泛部署也带来了巨大的安全风险，最引人注目的是迅速注入和越狱攻击的形式。   为了系统性地评估LLM漏洞--特别是外部即时注入--我们对八个商业模型进行了一系列实验。每个型号的测试都没有进行补充清理，仅依赖其内置的保障措施。结果暴露了可利用的弱点，并强调需要采取更强有力的安全措施。检查了四种类型的攻击：直接注射、间接（外部）注射、基于图像的注射和即时泄漏。比较分析表明Claude 3表现出相对更高的鲁棒性;然而，经验研究结果证实，为了实现可靠的保护，仍然需要额外的防御措施，例如输入正常化。



## **34. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

解码LLM中的潜在攻击：通过Web摘要中的HTML提示注入 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05831v1) [paper-pdf](http://arxiv.org/pdf/2509.05831v1)

**Authors**: Ishaan Verma

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.

摘要: 大型语言模型（LLM）越来越多地集成到基于Web的内容摘要系统中，但它们对即时注入攻击的敏感性仍然是一个紧迫的问题。在这项研究中，我们探索了如何利用非可见的HTML元素（例如<meta>、咏叹调标签和alt属性）来嵌入对抗性指令，而不改变网页的可见内容。我们引入了一个由280个静态网页组成的新颖数据集，平均分为干净和对抗注入版本，使用不同的基于HTML的策略制作。这些页面通过浏览器自动化管道进行处理，以提取原始HTML和渲染文本，密切模仿现实世界的LLM部署场景。我们评估了两个最先进的开源模型Llama 4 Scout（Meta）和Gemma 9 B IT（Google）总结此内容的能力。使用词汇（ROUGE-L）和语义（SBERT cos相似性）指标以及手动注释，我们评估这些隐蔽注入的影响。我们的研究结果显示，超过29%的注射样本导致Llama 4 Scout总结发生了显着变化，而Gemma 9 B IT的成功率较低，但并非微不足道，为15%。这些结果凸显了LLM驱动的网络管道中一个关键且在很大程度上被忽视的漏洞，其中隐藏的对抗内容可以巧妙地操纵模型输出。我们的工作为评估基于HTML的即时注入提供了一个可重复的框架和基准，并强调了涉及Web内容的LLM应用程序中对稳健的缓解策略的迫切需要。



## **35. Membership Inference Attacks on LLM-based Recommender Systems**

对基于LLM的推荐系统的成员推断攻击 cs.IR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.18665v2) [paper-pdf](http://arxiv.org/pdf/2508.18665v2)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.

摘要: 基于大型语言模型（LLM）的推荐系统（RecSys）可以灵活地调整推荐系统以适应不同的领域。它利用上下文学习（ICL），即提示来定制推荐功能，其中包括敏感的历史用户特定项目交互，例如，隐性反馈，例如点击的项目或明确的产品评论。此类私人信息可能会受到新型隐私攻击。然而，对于这个重要问题还没有进行任何研究。我们设计了四种成员资格推断攻击（MIA），旨在揭示受害者的历史互动是否被系统提示使用。它们是\r {直接询问、幻觉、相似性和中毒攻击}，每一种都利用了LLM或RecSys的独特功能。我们在三个用于开发ICL-LLM RecSys的LLM和两个著名的RecSys基准数据集上仔细评估了它们。结果证实，LLM RecSys上的MIA威胁是现实的：直接询问和中毒攻击显示出显着较高的攻击优势。我们还分析了影响这些攻击的因素，例如系统提示中的射击次数以及受害者在射击中的位置。



## **36. AntiDote: Bi-level Adversarial Training for Tamper-Resistant LLMs**

AntiDote：针对防篡改LLM的双级对抗培训 cs.CL

19 pages

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.08000v1) [paper-pdf](http://arxiv.org/pdf/2509.08000v1)

**Authors**: Debdeep Sanyal, Manodeep Ray, Murari Mandal

**Abstract**: The release of open-weight large language models (LLMs) creates a tension between advancing accessible research and preventing misuse, such as malicious fine-tuning to elicit harmful content. Current safety measures struggle to preserve the general capabilities of the LLM while resisting a determined adversary with full access to the model's weights and architecture, who can use full-parameter fine-tuning to erase existing safeguards. To address this, we introduce AntiDote, a bi-level optimization procedure for training LLMs to be resistant to such tampering. AntiDote involves an auxiliary adversary hypernetwork that learns to generate malicious Low-Rank Adaptation (LoRA) weights conditioned on the defender model's internal activations. The defender LLM is then trained with an objective to nullify the effect of these adversarial weight additions, forcing it to maintain its safety alignment. We validate this approach against a diverse suite of 52 red-teaming attacks, including jailbreak prompting, latent space manipulation, and direct weight-space attacks. AntiDote is upto 27.4\% more robust against adversarial attacks compared to both tamper-resistance and unlearning baselines. Crucially, this robustness is achieved with a minimal trade-off in utility, incurring a performance degradation of upto less than 0.5\% across capability benchmarks including MMLU, HellaSwag, and GSM8K. Our work offers a practical and compute efficient methodology for building open-weight models where safety is a more integral and resilient property.

摘要: 开放权重大型语言模型（LLM）的发布在推进可访问研究和防止滥用（例如恶意微调以引出有害内容）之间造成了紧张关系。当前的安全措施很难保留LLM的一般能力，同时抵御能够完全访问模型权重和架构的坚定对手，后者可以使用全参数微调来删除现有的保障措施。为了解决这个问题，我们引入了AntiDote，这是一种两级优化程序，用于训练LLM以抵抗此类篡改。AntiDote涉及一个辅助对手超网络，该网络根据防御者模型的内部激活来学习生成恶意的低等级适应（LoRA）权重。然后，防守者LLM接受训练，目标是抵消这些对抗性体重增加的影响，迫使其保持安全一致。我们针对一系列不同的52种红色团队攻击（包括越狱提示、潜在空间操纵和直接重量空间攻击）验证了这种方法。与抗篡改和取消学习基线相比，AntiDote对对抗性攻击的稳健性高出27.4%。至关重要的是，这种稳健性是在效用方面做出最小权衡的情况下实现的，导致MMLU、HellaSwag和GSM 8K等功能基准的性能下降高达不到0.5%。我们的工作提供了一个实用和计算效率高的方法来构建开放重量模型，其中安全性是一个更完整和弹性的属性。



## **37. Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System**

在基于LLM的开发系统中利用工具调用提示实现工具行为劫持 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05755v1) [paper-pdf](http://arxiv.org/pdf/2509.05755v1)

**Authors**: Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.

摘要: 基于法学硕士的代理系统利用大型语言模型来处理用户查询、做出决策并执行外部工具，以执行跨聊天机器人、客户服务和软件工程等领域的复杂任务。这些系统的一个关键组件是工具调用提示（TIP），它定义了工具交互协议并指导LLM确保工具使用的安全性和正确性。尽管TIP的安全性很重要，但在很大程度上被忽视了。这项工作调查了与TIP相关的安全风险，揭示了Cursor、Claude Code等基于LLM的主要系统容易受到远程代码执行（RCE）和拒绝服务（NOS）等攻击。通过系统性TIP利用工作流程（TEW），我们通过操纵工具调用演示了外部工具行为劫持。我们还提出了防御机制来增强基于LLM的代理系统中的TIP安全性。



## **38. Reasoning Introduces New Poisoning Attacks Yet Makes Them More Complicated**

推理引入了新的中毒攻击，但使它们变得更加复杂 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05739v1) [paper-pdf](http://arxiv.org/pdf/2509.05739v1)

**Authors**: Hanna Foerster, Ilia Shumailov, Yiren Zhao, Harsh Chaudhari, Jamie Hayes, Robert Mullins, Yarin Gal

**Abstract**: Early research into data poisoning attacks against Large Language Models (LLMs) demonstrated the ease with which backdoors could be injected. More recent LLMs add step-by-step reasoning, expanding the attack surface to include the intermediate chain-of-thought (CoT) and its inherent trait of decomposing problems into subproblems. Using these vectors for more stealthy poisoning, we introduce ``decomposed reasoning poison'', in which the attacker modifies only the reasoning path, leaving prompts and final answers clean, and splits the trigger across multiple, individually harmless components.   Fascinatingly, while it remains possible to inject these decomposed poisons, reliably activating them to change final answers (rather than just the CoT) is surprisingly difficult. This difficulty arises because the models can often recover from backdoors that are activated within their thought processes. Ultimately, it appears that an emergent form of backdoor robustness is originating from the reasoning capabilities of these advanced LLMs, as well as from the architectural separation between reasoning and final answer generation.

摘要: 对针对大型语言模型（LLM）的数据中毒攻击的早期研究证明了注入后门的容易性。最近的LLM添加了逐步推理，扩展了攻击面，以包括中间思想链（CoT）及其将问题分解为子问题的固有特征。使用这些载体进行更隐蔽的中毒，我们引入了“分解推理毒药”，其中攻击者仅修改推理路径，保留提示和最终答案干净，并将触发器拆分为多个单独无害的组件。   令人着迷的是，虽然仍然可以注射这些分解的毒药，但可靠地激活它们以改变最终答案（而不仅仅是CoT）却出奇地困难。出现这种困难是因为模型通常可以从其思维过程中激活的后门中恢复过来。最终，后门稳健性的一种新兴形式似乎源于这些高级LLM的推理能力，以及推理和最终答案生成之间的架构分离。



## **39. BadPromptFL: A Novel Backdoor Threat to Prompt-based Federated Learning in Multimodal Models**

BadminutFL：对多模式模型中基于预算的联邦学习的新型后门威胁 cs.LG

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.08040v3) [paper-pdf](http://arxiv.org/pdf/2508.08040v3)

**Authors**: Maozhen Zhang, Mengnan Zhao, Wei Wang, Bo Wang

**Abstract**: Prompt-based tuning has emerged as a lightweight alternative to full fine-tuning in large vision-language models, enabling efficient adaptation via learned contextual prompts. This paradigm has recently been extended to federated learning settings (e.g., PromptFL), where clients collaboratively train prompts under data privacy constraints. However, the security implications of prompt-based aggregation in federated multimodal learning remain largely unexplored, leaving a critical attack surface unaddressed. In this paper, we introduce \textbf{BadPromptFL}, the first backdoor attack targeting prompt-based federated learning in multimodal contrastive models. In BadPromptFL, compromised clients jointly optimize local backdoor triggers and prompt embeddings, injecting poisoned prompts into the global aggregation process. These prompts are then propagated to benign clients, enabling universal backdoor activation at inference without modifying model parameters. Leveraging the contextual learning behavior of CLIP-style architectures, BadPromptFL achieves high attack success rates (e.g., \(>90\%\)) with minimal visibility and limited client participation. Extensive experiments across multiple datasets and aggregation protocols validate the effectiveness, stealth, and generalizability of our attack, raising critical concerns about the robustness of prompt-based federated learning in real-world deployments.

摘要: 基于预算的调优已成为大型视觉语言模型中完全微调的轻量级替代方案，可以通过学习的上下文提示进行高效调整。该范式最近已扩展到联邦学习环境（例如，AtlantFL），客户在数据隐私限制下协作训练提示。然而，联邦多模式学习中基于预算的聚合的安全影响在很大程度上仍未得到探索，导致关键的攻击表面尚未得到解决。本文中，我们介绍了\textBF{BadoutFL}，这是第一个针对多模式对比模型中基于预算的联邦学习的后门攻击。在BadoutFL中，受影响的客户端联合优化本地后门触发器和提示嵌入，将有毒提示注入到全球聚合流程中。然后，这些提示被传播到良性客户端，从而在推理时启用通用后门，而无需修改模型参数。利用CLIP风格架构的上下文学习行为，BadminutFL实现了高攻击成功率（例如，\（>90\%\））可见性最低且客户参与有限。跨多个数据集和聚合协议的广泛实验验证了我们攻击的有效性、隐蔽性和可推广性，引发了人们对现实世界部署中基于预算的联邦学习稳健性的严重担忧。



## **40. Evo-MARL: Co-Evolutionary Multi-Agent Reinforcement Learning for Internalized Safety**

Evo-MARL：用于内部化安全的协同进化多智能体强化学习 cs.AI

accepted by the Trustworthy FMs workshop in ICCV 2025

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.03864v2) [paper-pdf](http://arxiv.org/pdf/2508.03864v2)

**Authors**: Zhenyu Pan, Yiting Zhang, Yutong Zhang, Jianshu Zhang, Haozheng Luo, Yuwei Han, Dennis Wu, Hong-Yu Chen, Philip S. Yu, Manling Li, Han Liu

**Abstract**: Multi-agent systems (MAS) built on multimodal large language models exhibit strong collaboration and performance. However, their growing openness and interaction complexity pose serious risks, notably jailbreak and adversarial attacks. Existing defenses typically rely on external guard modules, such as dedicated safety agents, to handle unsafe behaviors. Unfortunately, this paradigm faces two challenges: (1) standalone agents offer limited protection, and (2) their independence leads to single-point failure-if compromised, system-wide safety collapses. Naively increasing the number of guard agents further raises cost and complexity. To address these challenges, we propose Evo-MARL, a novel multi-agent reinforcement learning (MARL) framework that enables all task agents to jointly acquire defensive capabilities. Rather than relying on external safety modules, Evo-MARL trains each agent to simultaneously perform its primary function and resist adversarial threats, ensuring robustness without increasing system overhead or single-node failure. Furthermore, Evo-MARL integrates evolutionary search with parameter-sharing reinforcement learning to co-evolve attackers and defenders. This adversarial training paradigm internalizes safety mechanisms and continually enhances MAS performance under co-evolving threats. Experiments show that Evo-MARL reduces attack success rates by up to 22% while boosting accuracy by up to 5% on reasoning tasks-demonstrating that safety and utility can be jointly improved.

摘要: 基于多模式大型语言模型构建的多智能体系统（MAS）展现出强大的协作和性能。然而，它们日益增长的开放性和交互复杂性带来了严重的风险，特别是越狱和对抗性攻击。现有的防御系统通常依赖外部防护模块（例如专用安全代理）来处理不安全行为。不幸的是，这个范式面临着两个挑战：（1）独立代理提供的保护有限，（2）它们的独立性会导致单点故障--如果受到损害，系统范围的安全就会崩溃。天真地增加警卫特工的数量进一步增加了成本和复杂性。为了应对这些挑战，我们提出了Evo-MARL，这是一种新型的多智能体强化学习（MARL）框架，使所有任务智能体能够联合获得防御能力。Evo-MARL不是依赖外部安全模块，而是训练每个代理同时执行其主要功能并抵抗对抗威胁，在不增加系统负担或单节点故障的情况下确保稳健性。此外，Evo-MARL将进化搜索与参数共享强化学习集成，以共同进化攻击者和防御者。这种对抗性训练范式内化了安全机制，并在共同演变的威胁下不断增强MAS性能。实验表明，Evo-MARL将攻击成功率降低高达22%，同时将推理任务的准确性提高高达5%，证明安全性和实用性可以共同提高。



## **41. Differential Robustness in Transformer Language Models: Empirical Evaluation Under Adversarial Text Attacks**

Transformer语言模型的差分鲁棒性：对抗性文本攻击下的实证评估 cs.CR

8 pages, 4 tables, to appear in proceedings of Recent Advances in  Natural Language Processing (RANLP 2025) and ACL Anthology

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.09706v1) [paper-pdf](http://arxiv.org/pdf/2509.09706v1)

**Authors**: Taniya Gidatkar, Oluwaseun Ajao, Matthew Shardlow

**Abstract**: This study evaluates the resilience of large language models (LLMs) against adversarial attacks, specifically focusing on Flan-T5, BERT, and RoBERTa-Base. Using systematically designed adversarial tests through TextFooler and BERTAttack, we found significant variations in model robustness. RoBERTa-Base and FlanT5 demonstrated remarkable resilience, maintaining accuracy even when subjected to sophisticated attacks, with attack success rates of 0%. In contrast. BERT-Base showed considerable vulnerability, with TextFooler achieving a 93.75% success rate in reducing model accuracy from 48% to just 3%. Our research reveals that while certain LLMs have developed effective defensive mechanisms, these safeguards often require substantial computational resources. This study contributes to the understanding of LLM security by identifying existing strengths and weaknesses in current safeguarding approaches and proposes practical recommendations for developing more efficient and effective defensive strategies.

摘要: 这项研究评估了大型语言模型（LLM）对抗攻击的弹性，特别关注Flan-T5、BERT和RoBERTa-Base。通过TextFotify和BERTAttack使用系统设计的对抗测试，我们发现模型稳健性存在显着差异。RoBERTa-Base和FlanT 5表现出出色的韧性，即使在遭受复杂攻击时也能保持准确性，攻击成功率为0%。相比之下。BERT-Base表现出相当大的漏洞，TextFoilty在将模型准确性从48%降低到仅3%方面实现了93.75%的成功率。我们的研究表明，虽然某些LLM已经开发出了有效的防御机制，但这些保障措施通常需要大量的计算资源。本研究通过识别当前保障方法中现有的优点和缺点，有助于理解LLM安全，并为制定更高效和有效的防御策略提出实用建议。



## **42. Behind the Mask: Benchmarking Camouflaged Jailbreaks in Large Language Models**

面具背后：大型语言模型中伪装越狱的基准 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05471v1) [paper-pdf](http://arxiv.org/pdf/2509.05471v1)

**Authors**: Youjia Zheng, Mohammad Zandsalimy, Shanu Sushmita

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to a sophisticated form of adversarial prompting known as camouflaged jailbreaking. This method embeds malicious intent within seemingly benign language to evade existing safety mechanisms. Unlike overt attacks, these subtle prompts exploit contextual ambiguity and the flexible nature of language, posing significant challenges to current defense systems. This paper investigates the construction and impact of camouflaged jailbreak prompts, emphasizing their deceptive characteristics and the limitations of traditional keyword-based detection methods. We introduce a novel benchmark dataset, Camouflaged Jailbreak Prompts, containing 500 curated examples (400 harmful and 100 benign prompts) designed to rigorously stress-test LLM safety protocols. In addition, we propose a multi-faceted evaluation framework that measures harmfulness across seven dimensions: Safety Awareness, Technical Feasibility, Implementation Safeguards, Harmful Potential, Educational Value, Content Quality, and Compliance Score. Our findings reveal a stark contrast in LLM behavior: while models demonstrate high safety and content quality with benign inputs, they exhibit a significant decline in performance and safety when confronted with camouflaged jailbreak attempts. This disparity underscores a pervasive vulnerability, highlighting the urgent need for more nuanced and adaptive security strategies to ensure the responsible and robust deployment of LLMs in real-world applications.

摘要: 大型语言模型（LLM）越来越容易受到一种复杂形式的对抗激励，即所谓的虚拟越狱。这种方法将恶意意图嵌入看似良性的语言中，以逃避现有的安全机制。与公开攻击不同，这些微妙的提示利用了上下文模糊性和语言的灵活性，对当前的防御系统构成了重大挑战。本文研究了伪装越狱提示的结构和影响，强调了其欺骗性特征和传统基于关键词的检测方法的局限性。我们引入了一个新颖的基准数据集，即伪装的越狱提示，其中包含500个精心策划的示例（400个有害提示和100个良性提示），旨在对LLM安全协议进行严格的压力测试。此外，我们还提出了一个多方面的评估框架，从七个方面衡量危害性：安全意识、技术可行性、实施保障措施、潜在危害性、教育价值、内容质量和合规评分。我们的研究结果揭示了LLM行为的鲜明对比：虽然模型在良性输入下表现出高安全性和内容质量，但当面对伪装的越狱尝试时，它们的性能和安全性却显着下降。这种差异凸显了普遍存在的漏洞，凸显了迫切需要更加细致入微和适应性的安全策略，以确保在现实世界应用程序中负责任且稳健地部署LLM。



## **43. Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis**

神经面包屑：通过隐藏状态和注意力模式分析对LLM的成员推理攻击 cs.LG

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05449v1) [paper-pdf](http://arxiv.org/pdf/2509.05449v1)

**Authors**: Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah

**Abstract**: Membership inference attacks (MIAs) reveal whether specific data was used to train machine learning models, serving as important tools for privacy auditing and compliance assessment. Recent studies have reported that MIAs perform only marginally better than random guessing against large language models, suggesting that modern pre-training approaches with massive datasets may be free from privacy leakage risks. Our work offers a complementary perspective to these findings by exploring how examining LLMs' internal representations, rather than just their outputs, may provide additional insights into potential membership inference signals. Our framework, \emph{memTrace}, follows what we call \enquote{neural breadcrumbs} extracting informative signals from transformer hidden states and attention patterns as they process candidate sequences. By analyzing layer-wise representation dynamics, attention distribution characteristics, and cross-layer transition patterns, we detect potential memorization fingerprints that traditional loss-based approaches may not capture. This approach yields strong membership detection across several model families achieving average AUC scores of 0.85 on popular MIA benchmarks. Our findings suggest that internal model behaviors can reveal aspects of training data exposure even when output-based signals appear protected, highlighting the need for further research into membership privacy and the development of more robust privacy-preserving training techniques for large language models.

摘要: 成员资格推理攻击（MIA）揭示了特定数据是否用于训练机器学习模型，作为隐私审计和合规评估的重要工具。最近的研究报告称，MIA的表现仅比针对大型语言模型的随机猜测好一点点，这表明具有大量数据集的现代预训练方法可能没有隐私泄露风险。我们的工作通过探索如何检查LLM的内部表示而不仅仅是它们的输出来为这些发现提供了补充的视角。我们的框架，memTrace，遵循我们称之为神经面包屑的方法，在处理候选序列时，从Transformer隐藏状态和注意力模式中提取信息信号。通过分析逐层表示动态、注意力分布特征和跨层转变模式，我们检测传统基于损失的方法可能无法捕获的潜在记忆指纹。这种方法在多个模型系列中产生了强大的成员身份检测，在流行的MIA基准上实现了0.85的平均AUT分数。我们的研究结果表明，即使基于输出的信号看起来受到保护，内部模型行为也可以揭示训练数据暴露的各个方面，这凸显了对成员隐私进行进一步研究以及为大型语言模型开发更强大的隐私保护训练技术的必要性。



## **44. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection**

AgentArmor：对Agent DeliverTrace执行程序分析以防止即时注入 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2508.01249v2) [paper-pdf](http://arxiv.org/pdf/2508.01249v2)

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.

摘要: 大型语言模型（LLM）代理通过将自然语言推理与外部工具的执行相结合，提供了一个强大的新范式来解决各种问题。然而，它们的动态和不透明行为会带来严重的安全风险，特别是在存在即时注入攻击的情况下。在这项工作中，我们提出了一种新颖的见解，将代理运行时跟踪视为具有可分析语义的结构化程序。因此，我们提出了AgentArmor，这是一个程序分析框架，它将代理跟踪转换为基于图形中间表示的结构化程序依赖性表示（例如，CGM、DFG和PDG）并通过类型系统强制执行安全策略。AgentArmor由三个关键组件组成：（1）一个图形构造器，它将代理的运行时跟踪重建为基于图形的中间表示，其中描述了控制和数据流;（2）一个属性注册表，它附加了交互工具\&数据的安全相关元数据，以及（3）一个类型系统，它执行静态推理和检查中间表示。通过将代理行为表示为结构化程序，AgentArmor可以对敏感数据流、信任边界和策略违规进行程序分析。在AgentDojo基准测试中对AgentArmor进行了测试，结果表明，AgentArmor可以将ASR降低到3%，而效用下降仅为1%。



## **45. RINSER: Accurate API Prediction Using Masked Language Models**

RINser：使用掩蔽语言模型进行准确的API预测 cs.CY

16 pages, 8 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04887v1) [paper-pdf](http://arxiv.org/pdf/2509.04887v1)

**Authors**: Muhammad Ejaz Ahmed, Christopher Cody, Muhammad Ikram, Sean Lamont, Alsharif Abuadbba, Seyit Camtepe, Surya Nepal, Muhammad Ali Kaafar

**Abstract**: Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.

摘要: 恶意软件作者通常使用混淆将API身份隐藏在二进制文件中，这使得人类专家理解程序的行为和意图变得困难且耗时。自动API预测工具对于有效分析未知二进制文件来说是必要的，可以促进快速恶意软件分类，同时减少人类分析师的工作量。在本文中，我们介绍了RINBER（使用maSked languagE模型leRning的ACATER API预测），这是一个用于预测Windows API（WinAPI）函数名称的自动化框架。RINser引入了API代码印的新颖概念，即一组与API相关的汇编指令，并支持x86 PE二进制文件。RINser依赖BERT的掩蔽语言模型（LM）来大规模预测API名称，正常二进制文件的准确性达到85.77%，剥离二进制文件的准确性达到82.88%。我们在一个包含来自11，098个恶意软件二进制文件的470万个API代码的大型数据集上评估了RINser，涵盖了4，123个独特的Windows API，使其成为此类类型中最大的公开可用数据集。RINBER在我们的数据集中成功发现了65个与C2通信、间谍和规避相关的模糊Windows API，但商业反汇编器IDA未能识别这些API。此外，我们将RINBER与三种最先进的方法进行了比较，结果显示预测准确性提高了20%以上。我们还展示了RINBER对对抗攻击（包括指令随机化和代码置换）的弹性，性能下降不超过3%。



## **46. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

解药：微调后大型语言模型的安全调整，防止有害的微调 cs.AI

Rejected by AAAI25-AIA. Accepted by ICML25. Authors are thankful to  the anonymous reviewers from both AAAI25-AIA and ICML25

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2408.09600v3) [paper-pdf](http://arxiv.org/pdf/2408.09600v3)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks -- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. While several defenses have been proposed, our evaluation shows that existing defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks. Code is available at https://github.com/git-disl/Antidote.

摘要: 安全一致的大型语言模型（LLM）容易受到有害的微调攻击--微调数据集中混合的一些有害数据可能会破坏LLM的安全一致性。虽然已经提出了几种防御措施，但我们的评估表明，现有的防御措施会失败\textit{当选择一些特定的训练超参数时} --微调阶段中的大学习率或大量训练时期很容易使防御无效。为此，我们提出了Antidote，这是一种微调后的解决方案，它仍然是\textBF{\texttit {与微调阶段中的训练超参数不可知}。解药依赖于这样的理念：通过去除有害参数，有害模型可以从有害行为中恢复，无论这些有害参数是如何在微调阶段形成的。凭借这一理念，我们在有害微调后引入一次性修剪阶段，以删除导致有害内容生成的有害权重。尽管它的简单性令人尴尬，但经验结果表明，解毒剂可以降低有害评分，同时保持下游任务的准确性。代码可在https://github.com/git-disl/Antidote上获取。



## **47. Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs**

注意差距：使用行动图评估LLM中的模型和统计级别漏洞 cs.CL

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04802v1) [paper-pdf](http://arxiv.org/pdf/2509.04802v1)

**Authors**: Ilham Wicaksono, Zekun Wu, Theo King, Adriano Koshiyama, Philip Treleaven

**Abstract**: As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.

摘要: 随着大型语言模型向代理系统过渡，当前的安全评估框架在评估特定于部署的风险方面面临着严重差距。我们引入了AgentSeer，这是一个基于可观察性的评估框架，它将代理执行分解为粒度动作和组件图，从而实现系统性代理情景评估。通过使用HarmBench单轮攻击和迭代细化攻击对GPT-OSS-20 B和Gemini-2.0-Flash进行跨模型验证，我们展示了模型级和代理级漏洞配置文件之间的根本差异。模型级评估揭示了基线差异：GPT-OSS-20 B（39.47%ASB）与Gemini-2.0-Flash（50.00%ASB），两种模型都表现出对社会工程的敏感性，同时保持基于逻辑的攻击抵抗力。然而，代理层面的评估暴露了传统评估所看不到的代理特定风险。我们发现了仅在代理环境中出现的“仅代理”漏洞，工具调用显示两种模型的ASB高出24-60%。跨模型分析揭示了通用的代理模式、作为最高风险工具的代理传输操作、语义而非语法漏洞机制、取决于上下文的攻击有效性，以及绝对ASB级别的模型特定安全配置文件和最佳注入策略。从模型级到代理上下文的直接攻击转移显示出性能下降（GPT-OSS-20 B：57%人体注射ASO; Gemini-2.0-Flash：28%），而上下文感知迭代攻击成功地破坏了在模型级失败的目标，证实了系统性评估差距。这些发现确定了对主体情境评估范式的迫切需求，AgentSeer提供了标准化的方法论和经验验证。



## **48. Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs**

突破构建：用于保护LLM的基于预算的攻击威胁模型 cs.CL

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04615v1) [paper-pdf](http://arxiv.org/pdf/2509.04615v1)

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.

摘要: 大型语言模型（LLM）的激增带来了关键的安全挑战，对抗行为者可以操纵输入提示造成重大伤害并规避安全一致。这些基于预算的攻击利用模型设计、培训和上下文理解中的漏洞，导致知识产权盗窃、错误信息生成和用户信任度侵蚀。系统地了解这些攻击载体是开发稳健对策的基础步骤。本文对基于预算的攻击方法进行了全面的文献调查，对它们进行了分类，以提供明确的威胁模型。通过详细介绍这些漏洞利用的机制和影响，本调查旨在为研究界构建下一代安全LLM的努力提供信息，这些LLM本质上可以抵抗未经授权的提炼、微调和编辑。



## **49. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

Concept-ROT：使用模型编辑在大型语言模型中中毒概念 cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.13341v2) [paper-pdf](http://arxiv.org/pdf/2412.13341v2)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.

摘要: 模型编辑方法通过改变一小组有针对性的网络权重来修改大型语言模型的特定行为，并且需要很少的数据和计算。这些方法可用于恶意应用程序，例如插入错误信息或简单的特洛伊木马，当存在触发词时，这些木马会导致对手指定的行为。虽然之前的编辑方法专注于将单个单词与固定输出联系起来的相对受限的场景，但我们表明编辑技术可以以类似的效果集成更复杂的行为。我们开发了Concept-ROT，这是一种基于模型编辑的方法，可以有效地插入特洛伊木马，这些木马不仅表现出复杂的输出行为，而且还会触发高级概念--从而呈现出一种全新的特洛伊木马攻击。具体来说，我们将特洛伊木马插入到前沿安全调整的LLM中，这些LLM仅在存在“计算机科学”或“古代文明”等概念时才会触发。“当被触发时，特洛伊木马会越狱该模型，使其回答原本会拒绝的有害问题。我们的结果进一步引发了人们对机器学习模型木马攻击的实用性和潜在后果的担忧。



## **50. An Automated, Scalable Machine Learning Model Inversion Assessment Pipeline**

自动化、可扩展的机器学习模型反演评估管道 cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04214v1) [paper-pdf](http://arxiv.org/pdf/2509.04214v1)

**Authors**: Tyler Shumaker, Jessica Carpenter, David Saranchak, Nathaniel D. Bastian

**Abstract**: Machine learning (ML) models have the potential to transform military battlefields, presenting a large external pressure to rapidly incorporate them into operational settings. However, it is well-established that these ML models are vulnerable to a number of adversarial attacks throughout the model deployment pipeline that threaten to negate battlefield advantage. One broad category is privacy attacks (such as model inversion) where an adversary can reverse engineer information from the model, such as the sensitive data used in its training. The ability to quantify the risk of model inversion attacks (MIAs) is not well studied, and there is a lack of automated developmental test and evaluation (DT&E) tools and metrics to quantify the effectiveness of privacy loss of the MIA. The current DT&E process is difficult because ML model inversions can be hard for a human to interpret, subjective when they are interpretable, and difficult to quantify in terms of inversion quality. Additionally, scaling the DT&E process is challenging due to many ML model architectures and data modalities that need to be assessed. In this work, we present a novel DT&E tool that quantifies the risk of data privacy loss from MIAs and introduces four adversarial risk dimensions to quantify privacy loss. Our DT&E pipeline combines inversion with vision language models (VLMs) to improve effectiveness while enabling scalable analysis. We demonstrate effectiveness using multiple MIA techniques and VLMs configured for zero-shot classification and image captioning. We benchmark the pipeline using several state-of-the-art MIAs in the computer vision domain with an image classification task that is typical in military applications. In general, our innovative pipeline extends the current model inversion DT&E capabilities by improving the effectiveness and scalability of the privacy loss analysis in an automated fashion.

摘要: 机器学习（ML）模型有潜力改变军事战场，这给快速将其纳入作战环境带来了巨大的外部压力。然而，众所周知，这些ML模型在整个模型部署管道中容易受到许多对抗攻击，这些攻击可能会抵消战场优势。其中一个广泛的类别是隐私攻击（例如模型倒置），其中对手可以从模型中反向工程信息，例如训练中使用的敏感数据。量化模型倒置攻击（MIA）风险的能力尚未得到充分研究，并且缺乏自动化开发测试和评估（DT & E）工具和指标来量化MIA隐私损失的有效性。当前的DT & E过程很困难，因为ML模型倒置对人类来说可能很难解释，当它们可解释时是主观的，并且很难在倒置质量方面量化。此外，由于需要评估许多ML模型架构和数据模式，扩展DT & E流程具有挑战性。在这项工作中，我们提出了一种新颖的DT & E工具，该工具量化了MIA造成的数据隐私损失的风险，并引入了四个对抗风险维度来量化隐私损失。我们的DT & E管道将倒置与视觉语言模型（VLM）相结合，以提高有效性，同时实现可扩展分析。我们使用多种MIA技术和配置用于零镜头分类和图像字幕的VLM来证明有效性。我们使用计算机视觉领域的几种最先进的MIA对管道进行基准测试，并执行军事应用中典型的图像分类任务。总的来说，我们的创新管道通过以自动化方式提高隐私损失分析的有效性和可扩展性来扩展当前的模型倒置DT & E功能。



