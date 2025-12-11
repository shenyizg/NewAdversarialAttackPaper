# Latest Adversarial Attack Papers
**update at 2025-12-11 09:54:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improved Pseudorandom Codes from Permuted Puzzles**

从排列谜题中改进的伪随机码 cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08918v1) [paper-pdf](https://arxiv.org/pdf/2512.08918v1)

**Authors**: Miranda Christ, Noah Golowich, Sam Gunn, Ankur Moitra, Daniel Wichs

**Abstract**: Watermarks are an essential tool for identifying AI-generated content. Recently, Christ and Gunn (CRYPTO '24) introduced pseudorandom error-correcting codes (PRCs), which are equivalent to watermarks with strong robustness and quality guarantees. A PRC is a pseudorandom encryption scheme whose decryption algorithm tolerates a high rate of errors. Pseudorandomness ensures quality preservation of the watermark, and error tolerance of decryption translates to the watermark's ability to withstand modification of the content.   In the short time since the introduction of PRCs, several works (NeurIPS '24, RANDOM '25, STOC '25) have proposed new constructions. Curiously, all of these constructions are vulnerable to quasipolynomial-time distinguishing attacks. Furthermore, all lack robustness to edits over a constant-sized alphabet, which is necessary for a meaningfully robust LLM watermark. Lastly, they lack robustness to adversaries who know the watermarking detection key. Until now, it was not clear whether any of these properties was achievable individually, let alone together.   We construct pseudorandom codes that achieve all of the above: plausible subexponential pseudorandomness security, robustness to worst-case edits over a binary alphabet, and robustness against even computationally unbounded adversaries that have the detection key. Pseudorandomness rests on a new assumption that we formalize, the permuted codes conjecture, which states that a distribution of permuted noisy codewords is pseudorandom. We show that this conjecture is implied by the permuted puzzles conjecture used previously to construct doubly efficient private information retrieval. To give further evidence, we show that the conjecture holds against a broad class of simple distinguishers, including read-once branching programs.

摘要: 水印是识别AI生成内容的重要工具。最近，Christ和Gunn（2004年）引入了伪随机纠错码（PRCs），它相当于具有强鲁棒性和质量保证的水印。PRC是一种伪随机加密方案，其解密算法容忍高错误率。伪随机性确保水印的质量保持，并且解密的容错性转化为水印承受内容修改的能力。   自PRC引入以来的短时间内，几部作品（NeurIPS ' 24、RAN多姆' 25、STOC ' 25）提出了新的构造。奇怪的是，所有这些结构都容易受到准次数区分攻击。此外，所有这些都缺乏对恒定大小字母表的编辑的鲁棒性，而这对于有意义的鲁棒性LLM水印来说是必要的。最后，它们对知道水印检测关键的对手缺乏鲁棒性。到目前为止，尚不清楚这些性能中的任何一项是否可以单独实现，更不用说一起实现了。   我们构建伪随机码来实现上述所有功能：看似合理的亚指数伪随机安全性、对二进制字母表上最坏情况编辑的鲁棒性，以及对拥有检测密钥的计算无界对手的鲁棒性。伪随机性取决于我们形式化的一个新假设，即置换码猜想，它表明置换有噪代码字的分布是伪随机的。我们表明，之前用于构建双倍高效的私人信息检索的排列谜题猜想暗示了这个猜想。为了提供进一步的证据，我们表明该猜想适用于一类广泛的简单简化程序，包括一次读分支程序。



## **2. When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation**

当表泄露时：攻击基于LLM的表格数据生成中的字符串重新同步 cs.LG

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08875v1) [paper-pdf](https://arxiv.org/pdf/2512.08875v1)

**Authors**: Joshua Ward, Bochao Gu, Chi-Hua Wang, Guang Cheng

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data.

摘要: 大型语言模型（LLM）最近在生成高质量表格合成数据方面表现出色。在实践中，出现了两种用于使LLM适应表格数据生成的主要方法：（i）直接在表格数据集上微调较小的模型，以及（ii）通过上下文中提供的示例来提示较大的模型。在这项工作中，我们表明，这两种制度的流行实现都表现出通过从训练数据中复制记忆的数字模式来损害隐私的倾向。为了系统性地分析这种风险，我们引入了一种名为LevAtt的简单无箱成员推断攻击（MIA），该攻击假设仅对生成的合成数据进行对抗访问，并针对合成观察中的数字字符串序列。使用这种方法，我们的攻击暴露了广泛的模型和数据集中的大量隐私泄露，在某些情况下，甚至是最先进模型上的完美成员资格分类器。我们的研究结果强调了基于LLM的合成数据生成的独特隐私漏洞以及有效防御的必要性。为此，我们提出了两种方法，包括一种新颖的采样策略，可以在生成期间战略性地扰乱数字。我们的评估表明，这种方法可以在合成数据的保真度和实用性损失最小的情况下击败这些攻击。



## **3. Secure and Privacy-Preserving Federated Learning for Next-Generation Underground Mine Safety**

安全且保护隐私的联邦学习，以实现下一代地下矿山安全 cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08862v1) [paper-pdf](https://arxiv.org/pdf/2512.08862v1)

**Authors**: Mohamed Elmahallawy, Sanjay Madria, Samuel Frimpong

**Abstract**: Underground mining operations depend on sensor networks to monitor critical parameters such as temperature, gas concentration, and miner movement, enabling timely hazard detection and safety decisions. However, transmitting raw sensor data to a centralized server for machine learning (ML) model training raises serious privacy and security concerns. Federated Learning (FL) offers a promising alternative by enabling decentralized model training without exposing sensitive local data. Yet, applying FL in underground mining presents unique challenges: (i) Adversaries may eavesdrop on shared model updates to launch model inversion or membership inference attacks, compromising data privacy and operational safety; (ii) Non-IID data distributions across mines and sensor noise can hinder model convergence. To address these issues, we propose FedMining--a privacy-preserving FL framework tailored for underground mining. FedMining introduces two core innovations: (1) a Decentralized Functional Encryption (DFE) scheme that keeps local models encrypted, thwarting unauthorized access and inference attacks; and (2) a balancing aggregation mechanism to mitigate data heterogeneity and enhance convergence. Evaluations on real-world mining datasets demonstrate FedMining's ability to safeguard privacy while maintaining high model accuracy and achieving rapid convergence with reduced communication and computation overhead. These advantages make FedMining both secure and practical for real-time underground safety monitoring.

摘要: 地下采矿作业依赖传感器网络来监控温度、气体浓度和矿工移动等关键参数，从而能够及时检测危险并做出安全决策。然而，将原始传感器数据传输到集中式服务器进行机器学习（ML）模型训练会引发严重的隐私和安全问题。联邦学习（FL）通过在不暴露敏感的本地数据的情况下实现去中心化模型训练，提供了一种有前途的替代方案。然而，在地下采矿中应用FL带来了独特的挑战：（i）对手可能会窃听共享模型更新以发起模型倒置或隶属度推断攻击，从而损害数据隐私和操作安全性;（ii）矿山中的非IID数据分布和传感器噪音可能会阻碍模型收敛。为了解决这些问题，我们提出了FedMining--一个专为地下采矿量身定制的隐私保护FL框架。FedMining引入了两项核心创新：（1）去中心化功能加密（DTE）方案，该方案保持本地模型的加密，阻止未经授权的访问和推理攻击;（2）平衡聚合机制，以减轻数据的同质性并增强收敛。对现实世界挖掘数据集的评估表明，FedMining有能力保护隐私，同时保持高模型准确性并实现快速融合，减少通信和计算负担。这些优势使FedMining在实时地下安全监控中既安全又实用。



## **4. Developing Distance-Aware Uncertainty Quantification Methods in Physics-Guided Neural Networks for Reliable Bearing Health Prediction**

在物理引导神经网络中开发距离感知不确定性量化方法，以实现可靠的轴承健康预测 cs.LG

Under review at Structural health Monitoring - SAGE

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08499v1) [paper-pdf](https://arxiv.org/pdf/2512.08499v1)

**Authors**: Waleed Razzaq, Yun-Bo Zhao

**Abstract**: Accurate and uncertainty-aware degradation estimation is essential for predictive maintenance in safety-critical systems like rotating machinery with rolling-element bearings. Many existing uncertainty methods lack confidence calibration, are costly to run, are not distance-aware, and fail to generalize under out-of-distribution data. We introduce two distance-aware uncertainty methods for deterministic physics-guided neural networks: PG-SNGP, based on Spectral Normalization Gaussian Process, and PG-SNER, based on Deep Evidential Regression. We apply spectral normalization to the hidden layers so the network preserves distances from input to latent space. PG-SNGP replaces the final dense layer with a Gaussian Process layer for distance-sensitive uncertainty, while PG-SNER outputs Normal Inverse Gamma parameters to model uncertainty in a coherent probabilistic form. We assess performance using standard accuracy metrics and a new distance-aware metric based on the Pearson Correlation Coefficient, which measures how well predicted uncertainty tracks the distance between test and training samples. We also design a dynamic weighting scheme in the loss to balance data fidelity and physical consistency. We test our methods on rolling-element bearing degradation using the PRONOSTIA dataset and compare them with Monte Carlo and Deep Ensemble PGNNs. Results show that PG-SNGP and PG-SNER improve prediction accuracy, generalize reliably under OOD conditions, and remain robust to adversarial attacks and noise.

摘要: 准确且具有不确定性的退化估计对于具有滚动元件轴承的旋转机械等安全关键系统的预测性维护至关重要。许多现有的不确定性方法缺乏置信度校准、运行成本高、不具有距离意识，并且无法在非分布数据下进行概括。我们为确定性物理引导神经网络引入了两种距离感知不确定性方法：基于谱正规化高斯过程的PG-SNGP和基于深度证据回归的PG-SNER。我们对隐藏层应用光谱正规化，以便网络保留从输入到潜在空间的距离。PG-SNGP用高斯过程层取代最终的密集层，以实现距离敏感的不确定性，而PG-SNER输出正态逆伽玛参数，以连贯的概率形式对不确定性进行建模。我们使用标准准确性指标和基于皮尔逊相关系数的新距离感知指标来评估性能，皮尔逊相关系数衡量预测的不确定性跟踪测试和训练样本之间距离的程度。我们还设计了一个动态加权方案，以平衡数据保真度和物理一致性的损失。我们使用PRONOSTIA数据集测试我们关于滚动元件轴承退化的方法，并将其与Monte Carlo和Deep Ensemble PGNN进行比较。结果表明，PG-SNGP和PG-SNER提高了预测精度，在OOD条件下可靠地推广，并对对抗性攻击和噪声保持鲁棒性。



## **5. Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs**

防御LLM中的间接即时注入攻击只需注意力 cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium 2026

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08417v1) [paper-pdf](https://arxiv.org/pdf/2512.08417v1)

**Authors**: Yinan Zhong, Qianhao Miao, Yanjiao Chen, Jiangyi Deng, Yushi Cheng, Wenyuan Xu

**Abstract**: Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.

摘要: 大型语言模型（LLM）已集成到许多应用程序中（例如，Web代理）来执行更复杂的任务。然而，LLM授权的应用程序很容易受到间接提示注入（IPI）攻击，其中指令是通过不可信的外部数据源注入的。本文介绍了Rennervate，一个用于检测和防止IPI攻击的防御框架。Rennervate利用注意力功能来检测细粒度代币级别的隐蔽注入，从而实现精确的清理，以中和IPI攻击，同时维护LLM功能。具体来说，代币级检测器通过两步注意池机制实现，该机制聚集注意力头和响应代币以进行IPI检测和清理。此外，我们还建立了一个细粒度的IPI数据集FIPI，将其开源以支持进一步的研究。大量实验证实，Rennervate优于15种商业和学术IPI防御方法，在5个LLM和6个数据集上实现了高精度。我们还证明，Rennervate可以转移到不可见的攻击中，并且对适应性对手具有强大的鲁棒性。



## **6. Exposing and Defending Membership Leakage in Vulnerability Prediction Models**

漏洞预测模型中成员泄漏的暴露和防御 cs.CR

Accepted at APSEC 2025

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08291v1) [paper-pdf](https://arxiv.org/pdf/2512.08291v1)

**Authors**: Yihan Liao, Jacky Keung, Xiaoxue Ma, Jingyu Zhang, Yicheng Sun

**Abstract**: Neural models for vulnerability prediction (VP) have achieved impressive performance by learning from large-scale code repositories. However, their susceptibility to Membership Inference Attacks (MIAs), where adversaries aim to infer whether a particular code sample was used during training, poses serious privacy concerns. While MIA has been widely investigated in NLP and vision domains, its effects on security-critical code analysis tasks remain underexplored. In this work, we conduct the first comprehensive analysis of MIA on VP models, evaluating the attack success across various architectures (LSTM, BiGRU, and CodeBERT) and feature combinations, including embeddings, logits, loss, and confidence. Our threat model aligns with black-box and gray-box settings where prediction outputs are observable, allowing adversaries to infer membership by analyzing output discrepancies between training and non-training samples. The empirical findings reveal that logits and loss are the most informative and vulnerable outputs for membership leakage. Motivated by these observations, we propose a Noise-based Membership Inference Defense (NMID), which is a lightweight defense module that applies output masking and Gaussian noise injection to disrupt adversarial inference. Extensive experiments demonstrate that NMID significantly reduces MIA effectiveness, lowering the attack AUC from nearly 1.0 to below 0.65, while preserving the predictive utility of VP models. Our study highlights critical privacy risks in code analysis and offers actionable defense strategies for securing AI-powered software systems.

摘要: 漏洞预测（VP）的神经模型通过从大规模代码库中学习，取得了令人印象深刻的性能。然而，它们对会员推断攻击（MIA）的敏感性（对手旨在推断训练期间是否使用了特定的代码样本）带来了严重的隐私问题。虽然MIA在NLP和视觉领域得到了广泛的研究，但它对安全关键代码分析任务的影响仍然没有得到充分的研究。在这项工作中，我们对VP模型上的MIA进行了首次全面分析，评估了各种架构（LSTM、BiGRU和CodeBRT）和功能组合（包括嵌入、日志、损失和信心）之间的攻击成功率。我们的威胁模型与黑盒和灰盒设置保持一致，其中预测输出是可观察的，允许对手通过分析训练和非训练样本之间的输出差异来推断成员资格。实证结果显示，logits和损失是最具信息性和脆弱性的输出成员泄漏。受这些观察的启发，我们提出了一个基于噪声的成员资格推理防御（NMID），这是一个轻量级的防御模块，应用输出掩蔽和高斯噪声注入来破坏对抗性推理。大量的实验表明，NMID显著降低了MIA的有效性，将攻击AUC从近1.0降低到0.65以下，同时保留了VP模型的预测效用。我们的研究强调了代码分析中的关键隐私风险，并为保护人工智能驱动的软件系统提供了可操作的防御策略。



## **7. MIRAGE: Misleading Retrieval-Augmented Generation via Black-box and Query-agnostic Poisoning Attacks**

MIRST：通过黑匣子和查询不可知中毒攻击误导检索增强生成 cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08289v1) [paper-pdf](https://arxiv.org/pdf/2512.08289v1)

**Authors**: Tailun Chen, Yu He, Yan Wang, Shuo Shao, Haolun Zheng, Zhihao Liu, Jinfeng Li, Yuefeng Chen, Zhixuan Chu, Zhan Qin

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance LLMs with external knowledge but introduce a critical attack surface: corpus poisoning. While recent studies have demonstrated the potential of such attacks, they typically rely on impractical assumptions, such as white-box access or known user queries, thereby underestimating the difficulty of real-world exploitation. In this paper, we bridge this gap by proposing MIRAGE, a novel multi-stage poisoning pipeline designed for strict black-box and query-agnostic environments. Operating on surrogate model feedback, MIRAGE functions as an automated optimization framework that integrates three key mechanisms: it utilizes persona-driven query synthesis to approximate latent user search distributions, employs semantic anchoring to imperceptibly embed these intents for high retrieval visibility, and leverages an adversarial variant of Test-Time Preference Optimization (TPO) to maximize persuasion. To rigorously evaluate this threat, we construct a new benchmark derived from three long-form, domain-specific datasets. Extensive experiments demonstrate that MIRAGE significantly outperforms existing baselines in both attack efficacy and stealthiness, exhibiting remarkable transferability across diverse retriever-LLM configurations and highlighting the urgent need for robust defense strategies.

摘要: 检索增强生成（RAG）系统通过外部知识增强LLM，但引入了一个关键的攻击表面：主体中毒。虽然最近的研究已经证明了此类攻击的潜力，但它们通常依赖于不切实际的假设，例如白盒访问或已知用户查询，从而低估了现实世界利用的难度。在本文中，我们通过提出MISYS来弥合这一差距，MISYS是一种新型的多阶段中毒管道，专为严格的黑匣子和查询不可知的环境而设计。MISYS在代理模型反馈上运行，充当一个自动化优化框架，集成了三个关键机制：它利用角色驱动的查询合成来逼近潜在用户搜索分布，利用语义锚定以难以察觉的方式嵌入这些意图以实现高检索可见性，并利用测试时间偏好优化（LPO）的对抗变体来最大化说服力。为了严格评估这一威胁，我们构建了一个新的基准，该基准源自三个长篇、特定于领域的数据集。大量实验表明，MISYS在攻击功效和隐蔽性方面都显着优于现有基线，在不同的检索器LLM配置中表现出出色的可移植性，并凸显了对稳健防御策略的迫切需求。



## **8. Evaluating Vulnerabilities of Connected Vehicles Under Cyber Attacks by Attack-Defense Tree**

利用攻击-防御树评估互联汽车在网络攻击下的脆弱性 cs.CR

6 Pages, International Conference on Computing, Networking and Communication (ICNC), Maui, Hawaii, USA, 2026

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08204v1) [paper-pdf](https://arxiv.org/pdf/2512.08204v1)

**Authors**: Muhammad Baqer Mollah, Honggang Wang, Hua Fang

**Abstract**: Connected vehicles represent a key enabler of intelligent transportation systems, where vehicles are equipped with advanced communication, sensing, and computing technologies to interact not only with one another but also with surrounding infrastructures and the environment. Through continuous data exchange, such vehicles are capable of enhancing road safety, improving traffic efficiency, and ensuring more reliable mobility services. Further, when these capabilities are integrated with advanced automation technologies, the concept essentially evolves into connected and autonomous vehicles (CAVs). While connected vehicles primarily focus on seamless information sharing, autonomous vehicles are mainly dependent on advanced perception, decision-making, and control mechanisms to operate with minimal or without human intervention. However, as a result of connectivity, an adversary with malicious intentions might be able to compromise successfully by breaching the system components of CAVs. In this paper, we present an attack-tree based methodology for evaluating cyber security vulnerabilities in CAVs. In particular, we utilize the attack-defense tree formulation to systematically assess attack-leaf vulnerabilities, and before analyzing the vulnerability indices, we also define a measure of vulnerabilities, which is based on existing cyber security threats and corresponding defensive countermeasures.

摘要: 互联车辆是智能交通系统的关键推动者，其中车辆配备了先进的通信、传感和计算技术，不仅可以相互互动，还可以与周围的基础设施和环境互动。通过持续的数据交换，此类车辆能够增强道路安全、提高交通效率并确保更可靠的移动服务。此外，当这些功能与先进的自动化技术集成时，该概念本质上会演变为互联和自动驾驶汽车（CAB）。虽然互联汽车主要关注无缝信息共享，但自动驾驶汽车主要依赖于先进的感知、决策和控制机制，以便在最少或没有人为干预的情况下运行。然而，由于连接性，具有恶意意图的对手可能能够通过破坏卡韦的系统组件来成功妥协。在本文中，我们提出了一种基于攻击树的方法来评估卡韦中的网络安全漏洞。特别是，我们利用攻击-防御树公式来系统地评估攻击叶漏洞，在分析漏洞指数之前，我们还定义了漏洞衡量标准，该衡量标准基于现有的网络安全威胁和相应的防御对策。



## **9. A Practical Framework for Evaluating Medical AI Security: Reproducible Assessment of Jailbreaking and Privacy Vulnerabilities Across Clinical Specialties**

评估医疗人工智能安全性的实用框架：跨临床专业越狱和隐私漏洞的可重复性评估 cs.CR

6 pages, 1 figure, framework proposal

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08185v1) [paper-pdf](https://arxiv.org/pdf/2512.08185v1)

**Authors**: Jinghao Wang, Ping Zhang, Carter Yagemann

**Abstract**: Medical Large Language Models (LLMs) are increasingly deployed for clinical decision support across diverse specialties, yet systematic evaluation of their robustness to adversarial misuse and privacy leakage remains inaccessible to most researchers. Existing security benchmarks require GPU clusters, commercial API access, or protected health data -- barriers that limit community participation in this critical research area. We propose a practical, fully reproducible framework for evaluating medical AI security under realistic resource constraints. Our framework design covers multiple medical specialties stratified by clinical risk -- from high-risk domains such as emergency medicine and psychiatry to general practice -- addressing jailbreaking attacks (role-playing, authority impersonation, multi-turn manipulation) and privacy extraction attacks. All evaluation utilizes synthetic patient records requiring no IRB approval. The framework is designed to run entirely on consumer CPU hardware using freely available models, eliminating cost barriers. We present the framework specification including threat models, data generation methodology, evaluation protocols, and scoring rubrics. This proposal establishes a foundation for comparative security assessment of medical-specialist models and defense mechanisms, advancing the broader goal of ensuring safe and trustworthy medical AI systems.

摘要: 医学大型语言模型（LLM）越来越多地被部署用于不同专业的临床决策支持，但大多数研究人员仍然无法对其对抗性滥用和隐私泄露的稳健性进行系统评估。现有的安全基准测试需要图形处理器集群、商业API访问或受保护的健康数据--这些障碍限制了社区参与这一关键研究领域。我们提出了一个实用、完全可重复的框架，用于在现实资源限制下评估医疗人工智能安全性。我们的框架设计涵盖了按临床风险分层的多个医学专业--从急诊医学和精神病学等高风险领域到全科医学--解决越狱攻击（角色扮演、权威模仿、多回合操纵）和隐私提取攻击。所有评估均使用合成患者记录，无需获得机构审核委员会批准。该框架旨在完全在使用免费型号的消费者中央处理器硬件上运行，消除了成本障碍。我们提出了框架规范，包括威胁模型、数据生成方法、评估协议和评分规则。该提案为医疗专家模型和防御机制的比较安全评估奠定了基础，推进确保医疗人工智能系统安全可信的更广泛目标。



## **10. A Dynamic Coding Scheme to Prevent Covert Cyber-Attacks in Cyber-Physical Systems**

防止网络物理系统中隐蔽网络攻击的动态编码方案 eess.SY

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08134v1) [paper-pdf](https://arxiv.org/pdf/2512.08134v1)

**Authors**: Mahdi Taheri, Khashayar Khorasani, Nader Meskin

**Abstract**: In this paper, we address two main problems in the context of covert cyber-attacks in cyber-physical systems (CPS). First, we aim to investigate and develop necessary and sufficient conditions in terms of disruption resources of the CPS that enable adversaries to execute covert cyber-attacks. These conditions can be utilized to identify the input and output communication channels that are needed by adversaries to execute these attacks. Second, this paper introduces and develops a dynamic coding scheme as a countermeasure against covert cyber-attacks. Under certain conditions and assuming the existence of one secure input and two secure output communication channels, the proposed dynamic coding scheme prevents adversaries from executing covert cyber-attacks. A numerical case study of a flight control system is provided to demonstrate the capabilities of our proposed and developed dynamic coding scheme.

摘要: 在本文中，我们讨论了网络物理系统（CPS）中隐蔽网络攻击背景下的两个主要问题。首先，我们的目标是调查并制定CPS破坏资源方面的必要和充分的条件，使对手能够实施秘密网络攻击。这些条件可用于识别对手执行这些攻击所需的输入和输出通信通道。其次，本文引入并开发了一种动态编码方案作为对抗隐蔽网络攻击的对策。在一定条件下，假设存在一个安全输入和两个安全输出通信通道，提出的动态编码方案可以防止对手实施隐蔽的网络攻击。提供了飞行控制系统的数字案例研究，以证明我们提出和开发的动态编码方案的能力。



## **11. Universal Adversarial Suffixes Using Calibrated Gumbel-Softmax Relaxation**

使用校准的Gumbel-Softmax松弛的通用对抗后缀 cs.CL

10 pages

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08123v1) [paper-pdf](https://arxiv.org/pdf/2512.08123v1)

**Authors**: Sampriti Soor, Suklav Ghosh, Arijit Sur

**Abstract**: Language models (LMs) are often used as zero-shot or few-shot classifiers by scoring label words, but they remain fragile to adversarial prompts. Prior work typically optimizes task- or model-specific triggers, making results difficult to compare and limiting transferability. We study universal adversarial suffixes: short token sequences (4-10 tokens) that, when appended to any input, broadly reduce accuracy across tasks and models. Our approach learns the suffix in a differentiable "soft" form using Gumbel-Softmax relaxation and then discretizes it for inference. Training maximizes calibrated cross-entropy on the label region while masking gold tokens to prevent trivial leakage, with entropy regularization to avoid collapse. A single suffix trained on one model transfers effectively to others, consistently lowering both accuracy and calibrated confidence. Experiments on sentiment analysis, natural language inference, paraphrase detection, commonsense QA, and physical reasoning with Qwen2-1.5B, Phi-1.5, and TinyLlama-1.1B demonstrate consistent attack effectiveness and transfer across tasks and model families.

摘要: 语言模型（LM）通常被用作零镜头或少数镜头分类器，通过对标签词进行评分，但它们仍然很脆弱。以前的工作通常优化特定于任务或模型的触发器，使结果难以比较，并限制了可移植性。我们研究了通用的对抗性后缀：短令牌序列（4-10个令牌），当附加到任何输入时，会大大降低任务和模型的准确性。我们的方法使用Gumbel-Softmax松弛以可微的“软”形式学习后缀，然后将其离散化以进行推理。训练最大化了标签区域上的校准交叉信息，同时屏蔽黄金代币以防止微不足道的泄露，并通过信息规范化以避免崩溃。在一个模型上训练的单个后缀有效地转移到其他模型上，从而持续降低准确性和校准置信度。使用Qwen 2 -1.5B、Phi-1.5和TinyLlama-1.1B进行的情感分析、自然语言推理、重述检测、常识QA和物理推理实验证明了一致的攻击有效性和跨任务和模型系列的转移。



## **12. Detecting Ambiguity Aversion in Cyberattack Behavior to Inform Cognitive Defense Strategies**

检测网络攻击行为中的模糊厌恶以告知认知防御策略 cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.08107v1) [paper-pdf](https://arxiv.org/pdf/2512.08107v1)

**Authors**: Stephan Carney, Soham Hans, Sofia Hirschmann, Stacey Marsella, Yvonne Fonken, Peggy Wu, Nikolos Gurney

**Abstract**: Adversaries (hackers) attempting to infiltrate networks frequently face uncertainty in their operational environments. This research explores the ability to model and detect when they exhibit ambiguity aversion, a cognitive bias reflecting a preference for known (versus unknown) probabilities. We introduce a novel methodological framework that (1) leverages rich, multi-modal data from human-subjects red-team experiments, (2) employs a large language model (LLM) pipeline to parse unstructured logs into MITRE ATT&CK-mapped action sequences, and (3) applies a new computational model to infer an attacker's ambiguity aversion level in near-real time. By operationalizing this cognitive trait, our work provides a foundational component for developing adaptive cognitive defense strategies.

摘要: 试图渗透网络的对手（黑客）经常面临其操作环境的不确定性。这项研究探索了建模和检测他们何时表现出歧义厌恶的能力，这是一种反映对已知（与未知）概率偏好的认知偏见。我们引入了一种新颖的方法论框架，（1）利用来自人类受试者红队实验的丰富、多模式数据，（2）采用大型语言模型（LLM）管道将非结构化日志解析为MITRE ATA和CK映射的动作序列，（3）应用新的计算模型来近实时地推断攻击者的歧义厌恶水平。通过操作这种认知特征，我们的工作为开发适应性认知防御策略提供了基础组成部分。



## **13. An Adaptive Multi-Layered Honeynet Architecture for Threat Behavior Analysis via Deep Learning**

通过深度学习进行威胁行为分析的自适应多层蜜网架构 cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07827v1) [paper-pdf](https://arxiv.org/pdf/2512.07827v1)

**Authors**: Lukas Johannes Möller

**Abstract**: The escalating sophistication and variety of cyber threats have rendered static honeypots inadequate, necessitating adaptive, intelligence-driven deception. In this work, ADLAH is introduced: an Adaptive Deep Learning Anomaly Detection Honeynet designed to maximize high-fidelity threat intelligence while minimizing cost through autonomous orchestration of infrastructure. The principal contribution is offered as an end-to-end architectural blueprint and vision for an AI-driven deception platform. Feasibility is evidenced by a functional prototype of the central decision mechanism, in which a reinforcement learning (RL) agent determines, in real time, when sessions should be escalated from low-interaction sensor nodes to dynamically provisioned, high-interaction honeypots. Because sufficient live data were unavailable, field-scale validation is not claimed; instead, design trade-offs and limitations are detailed, and a rigorous roadmap toward empirical evaluation at scale is provided. Beyond selective escalation and anomaly detection, the architecture pursues automated extraction, clustering, and versioning of bot attack chains, a core capability motivated by the empirical observation that exposed services are dominated by automated traffic. Together, these elements delineate a practical path toward cost-efficient capture of high-value adversary behavior, systematic bot versioning, and the production of actionable threat intelligence.

摘要: 不断升级的复杂性和网络威胁的多样性使静态蜜罐变得不充分，需要自适应的、情报驱动的欺骗。在这项工作中，引入了ADLAH：一个自适应深度学习异常检测蜜网，旨在最大限度地提高高保真威胁情报，同时通过基础设施的自主编排最大限度地降低成本。主要贡献是作为人工智能驱动欺骗平台的端到端架构蓝图和愿景提供的。中央决策机制的功能原型证明了可行性，其中强化学习（RL）代理实时确定何时应该将会话从低交互性传感器节点升级到动态配置的高交互性蜜罐。由于无法获得足够的实时数据，因此没有要求进行现场规模验证;相反，详细介绍了设计权衡和限制，并提供了大规模经验评估的严格路线图。除了选择性升级和异常检测之外，该架构还追求机器人攻击链的自动化提取、集群和版本化，这是一项核心功能，其动机是基于经验观察，即暴露的服务由自动流量主导。这些要素共同描绘了一条以经济高效的方式捕获高价值对手行为、系统性机器人版本控制和产生可操作的威胁情报的实用路径。



## **14. Optimization-Guided Diffusion for Interactive Scene Generation**

交互式场景生成的优化引导扩散 cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07661v1) [paper-pdf](https://arxiv.org/pdf/2512.07661v1)

**Authors**: Shiaho Li, Naisheng Ye, Tianyu Li, Kashyap Chitta, Tuo An, Peng Su, Boyang Wang, Haiou Liu, Chen Lv, Hongyang Li

**Abstract**: Realistic and diverse multi-agent driving scenes are crucial for evaluating autonomous vehicles, but safety-critical events which are essential for this task are rare and underrepresented in driving datasets. Data-driven scene generation offers a low-cost alternative by synthesizing complex traffic behaviors from existing driving logs. However, existing models often lack controllability or yield samples that violate physical or social constraints, limiting their usability. We present OMEGA, an optimization-guided, training-free framework that enforces structural consistency and interaction awareness during diffusion-based sampling from a scene generation model. OMEGA re-anchors each reverse diffusion step via constrained optimization, steering the generation towards physically plausible and behaviorally coherent trajectories. Building on this framework, we formulate ego-attacker interactions as a game-theoretic optimization in the distribution space, approximating Nash equilibria to generate realistic, safety-critical adversarial scenarios. Experiments on nuPlan and Waymo show that OMEGA improves generation realism, consistency, and controllability, increasing the ratio of physically and behaviorally valid scenes from 32.35% to 72.27% for free exploration capabilities, and from 11% to 80% for controllability-focused generation. Our approach can also generate $5\times$ more near-collision frames with a time-to-collision under three seconds while maintaining the overall scene realism.

摘要: 真实且多样化的多智能体驾驶场景对于评估自动驾驶汽车至关重要，但对这项任务至关重要的安全关键事件在驾驶数据集中很少且代表性不足。数据驱动的场景生成通过从现有驾驶日志合成复杂的交通行为来提供低成本的替代方案。然而，现有模型通常缺乏可控制性或产生违反物理或社会约束的样本，从而限制了其可用性。我们提出了OMEK，这是一个优化引导、免训练的框架，它在场景生成模型的基于扩散的采样期间强制执行结构一致性和交互意识。OMEK通过约束优化重新锚定每个反向扩散步骤，引导一代走向物理上合理且行为上一致的轨迹。基于这个框架，我们将自我攻击者的相互作用制定为分布空间中的博弈论优化，逼近纳什均衡以生成现实的、对安全至关重要的对抗场景。在nuPlan和Waymo上的实验表明，OMEK提高了世代的真实感、一致性和可控性，将物理和行为有效场景的比例从32.35%提高到72.27%（对于免费探索能力），从11%提高到80%（对于以可开发性为重点的世代）。我们的方法还可以生成5美元以上的近碰撞帧，碰撞时间在三秒以内，同时保持整体场景的真实感。



## **15. Towards Robust DeepFake Detection under Unstable Face Sequences: Adaptive Sparse Graph Embedding with Order-Free Representation and Explicit Laplacian Spectral Prior**

在不稳定人脸序列下实现稳健的DeepFake检测：嵌入无序表示和显式拉普拉斯谱先验的自适应稀疏图 cs.CV

16 pages (including appendix)

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07498v1) [paper-pdf](https://arxiv.org/pdf/2512.07498v1)

**Authors**: Chih-Chung Hsu, Shao-Ning Chen, Chia-Ming Lee, Yi-Fang Wang, Yi-Shiuan Chou

**Abstract**: Ensuring the authenticity of video content remains challenging as DeepFake generation becomes increasingly realistic and robust against detection. Most existing detectors implicitly assume temporally consistent and clean facial sequences, an assumption that rarely holds in real-world scenarios where compression artifacts, occlusions, and adversarial attacks destabilize face detection and often lead to invalid or misdetected faces. To address these challenges, we propose a Laplacian-Regularized Graph Convolutional Network (LR-GCN) that robustly detects DeepFakes from noisy or unordered face sequences, while being trained only on clean facial data. Our method constructs an Order-Free Temporal Graph Embedding (OF-TGE) that organizes frame-wise CNN features into an adaptive sparse graph based on semantic affinities. Unlike traditional methods constrained by strict temporal continuity, OF-TGE captures intrinsic feature consistency across frames, making it resilient to shuffled, missing, or heavily corrupted inputs. We further impose a dual-level sparsity mechanism on both graph structure and node features to suppress the influence of invalid faces. Crucially, we introduce an explicit Graph Laplacian Spectral Prior that acts as a high-pass operator in the graph spectral domain, highlighting structural anomalies and forgery artifacts, which are then consolidated by a low-pass GCN aggregation. This sequential design effectively realizes a task-driven spectral band-pass mechanism that suppresses background information and random noise while preserving manipulation cues. Extensive experiments on FF++, Celeb-DFv2, and DFDC demonstrate that LR-GCN achieves state-of-the-art performance and significantly improved robustness under severe global and local disruptions, including missing faces, occlusions, and adversarially perturbed face detections.

摘要: 随着DeepFake一代变得越来越真实且抗检测，确保视频内容的真实性仍然具有挑战性。大多数现有的检测器隐含地假设时间一致且干净的面部序列，这一假设在现实世界场景中很少成立，因为现实世界场景中压缩伪影、遮挡和对抗攻击会破坏面部检测并经常导致无效或误判面部。为了解决这些挑战，我们提出了一种拉普拉斯-正规图卷积网络（LR-GCN），它可以从有噪或无序的面部序列中稳健地检测DeepFakes，同时仅在干净的面部数据上进行训练。我们的方法构建了一个无序时态图嵌入（OF-TGE），它根据语义亲和力将逐帧CNN特征组织到自适应稀疏图中。与受严格时间连续性约束的传统方法不同，OF-TGE捕获帧之间的固有特征一致性，使其能够对洗牌、缺失或严重损坏的输入具有弹性。我们进一步在图结构和节点特征上引入了双层稀疏机制，以抑制无效面孔的影响。至关重要的是，我们引入了一个显式的图拉普拉斯谱先验，它充当图谱域中的高通运算符，突出显示结构异常和伪造伪影，然后通过低通GCN聚合合并这些伪影。这种顺序设计有效地实现了任务驱动的光谱带通机制，该机制可以抑制背景信息和随机噪音，同时保留操纵线索。对FF++、Celeb-DFv 2和DFDC的大量实验表明，LR-GCN在严重的全球和局部干扰（包括人脸缺失、遮挡和对抗干扰的人脸检测）下实现了最先进的性能和显着提高的鲁棒性。



## **16. Pay Less Attention to Function Words for Free Robustness of Vision-Language Models**

为了视觉语言模型的自由鲁棒性，减少对虚词的关注 cs.LG

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.07222v2) [paper-pdf](https://arxiv.org/pdf/2512.07222v2)

**Authors**: Qiwei Tian, Chenhao Lin, Zhengyu Zhao, Chao Shen

**Abstract**: To address the trade-off between robustness and performance for robust VLM, we observe that function words could incur vulnerability of VLMs against cross-modal adversarial attacks, and propose Function-word De-Attention (FDA) accordingly to mitigate the impact of function words. Similar to differential amplifiers, our FDA calculates the original and the function-word cross-attention within attention heads, and differentially subtracts the latter from the former for more aligned and robust VLMs. Comprehensive experiments include 2 SOTA baselines under 6 different attacks on 2 downstream tasks, 3 datasets, and 3 models. Overall, our FDA yields an average 18/13/53% ASR drop with only 0.2/0.3/0.6% performance drops on the 3 tested models on retrieval, and a 90% ASR drop with a 0.3% performance gain on visual grounding. We demonstrate the scalability, generalization, and zero-shot performance of FDA experimentally, as well as in-depth ablation studies and analysis. Code will be made publicly at https://github.com/michaeltian108/FDA.

摘要: 为了解决鲁棒VLM的鲁棒性和性能之间的权衡，我们观察到功能字可能会导致VLM对跨模式对抗攻击的脆弱性，并相应地提出功能字去注意力（FDA）来减轻功能字的影响。与差异放大器类似，我们的FDA计算注意力头内的原始和功能字交叉注意力，并从前者中以差异方式减去后者，以获得更一致和更强大的VLM。全面的实验包括2个SOTA基线，在对2个下游任务、3个数据集和3个模型的6种不同攻击下。总体而言，我们的FDA在检索方面得出的ASB平均下降了18/13/53%，而3个测试型号的性能仅下降0.2/0.3/0.6%，而ASC下降了90%，视觉基础性能提高了0.3%。我们通过实验证明了FDA的可扩展性、通用性和零发射性能，以及深入的消融研究和分析。代码将在https://github.com/michaeltian108/FDA上公开发布。



## **17. ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking**

ThinkTrap：通过无限思维对黑匣子LLM服务进行拒绝服务攻击 cs.CR

This version includes the final camera-ready manuscript accepted by NDSS 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07086v1) [paper-pdf](https://arxiv.org/pdf/2512.07086v1)

**Authors**: Yunzhe Li, Jianan Wang, Hongzi Zhu, James Lin, Shan Chang, Minyi Guo

**Abstract**: Large Language Models (LLMs) have become foundational components in a wide range of applications, including natural language understanding and generation, embodied intelligence, and scientific discovery. As their computational requirements continue to grow, these models are increasingly deployed as cloud-based services, allowing users to access powerful LLMs via the Internet. However, this deployment model introduces a new class of threat: denial-of-service (DoS) attacks via unbounded reasoning, where adversaries craft specially designed inputs that cause the model to enter excessively long or infinite generation loops. These attacks can exhaust backend compute resources, degrading or denying service to legitimate users. To mitigate such risks, many LLM providers adopt a closed-source, black-box setting to obscure model internals. In this paper, we propose ThinkTrap, a novel input-space optimization framework for DoS attacks against LLM services even in black-box environments. The core idea of ThinkTrap is to first map discrete tokens into a continuous embedding space, then undertake efficient black-box optimization in a low-dimensional subspace exploiting input sparsity. The goal of this optimization is to identify adversarial prompts that induce extended or non-terminating generation across several state-of-the-art LLMs, achieving DoS with minimal token overhead. We evaluate the proposed attack across multiple commercial, closed-source LLM services. Our results demonstrate that, even far under the restrictive request frequency limits commonly enforced by these platforms, typically capped at ten requests per minute (10 RPM), the attack can degrade service throughput to as low as 1% of its original capacity, and in some cases, induce complete service failure.

摘要: 大型语言模型（LLM）已经成为广泛应用的基础组件，包括自然语言理解和生成，体现智能和科学发现。随着计算需求的不断增长，这些模型越来越多地部署为基于云的服务，允许用户通过互联网访问强大的LLM。然而，这种部署模型引入了一类新的威胁：通过无限推理的拒绝服务（DoS）攻击，其中攻击者精心设计了特别设计的输入，导致模型进入过长或无限的生成循环。这些攻击可能耗尽后端计算资源，降低或拒绝向合法用户提供服务。为了降低此类风险，许多LLM提供商采用闭源、黑匣子设置来掩盖模型内部内容。在本文中，我们提出了ThinkTrap，这是一种新型的输入空间优化框架，即使在黑匣子环境中也可以对LLM服务进行NOS攻击。ThinkTrap的核心思想是首先将离散令牌映射到连续嵌入空间，然后在利用输入稀疏性的低维子空间中进行高效的黑匣子优化。此优化的目标是识别在多个最先进的LLM之间引发扩展或非终止生成的对抗提示，以最小的令牌负载实现拒绝服务。我们评估了跨多个商业、闭源LLM服务的拟议攻击。我们的结果表明，即使远低于这些平台通常强制执行的限制性请求频率限制（通常限制为每分钟10个请求（10转/分钟）），攻击也可以将服务吞吐量降低至低至其原始容量的1%，在某些情况下，会导致完全的服务故障。



## **18. Replicating TEMPEST at Scale: Multi-Turn Adversarial Attacks Against Trillion-Parameter Frontier Models**

大规模复制TEMPST：针对万亿参数前沿模型的多轮对抗攻击 cs.CL

30 pages, 11 figures, 5 tables. Code and data: https://github.com/ricyoung/tempest-replication

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07059v1) [paper-pdf](https://arxiv.org/pdf/2512.07059v1)

**Authors**: Richard Young

**Abstract**: Despite substantial investment in safety alignment, the vulnerability of large language models to sophisticated multi-turn adversarial attacks remains poorly characterized, and whether model scale or inference mode affects robustness is unknown. This study employed the TEMPEST multi-turn attack framework to evaluate ten frontier models from eight vendors across 1,000 harmful behaviors, generating over 97,000 API queries across adversarial conversations with automated evaluation by independent safety classifiers. Results demonstrated a spectrum of vulnerability: six models achieved 96% to 100% attack success rate (ASR), while four showed meaningful resistance, with ASR ranging from 42% to 78%; enabling extended reasoning on identical architecture reduced ASR from 97% to 42%. These findings indicate that safety alignment quality varies substantially across vendors, that model scale does not predict adversarial robustness, and that thinking mode provides a deployable safety enhancement. Collectively, this work establishes that current alignment techniques remain fundamentally vulnerable to adaptive multi-turn attacks regardless of model scale, while identifying deliberative inference as a promising defense direction.

摘要: 尽管在安全对齐方面投入了大量资金，但大型语言模型对复杂多轮对抗攻击的脆弱性仍然很难描述，并且模型规模或推理模式是否会影响稳健性尚不清楚。这项研究采用TEMPEST多回合攻击框架来评估来自8家供应商的10个前沿模型，涵盖1，000种有害行为，通过独立安全分类器的自动评估，在对抗性对话中生成超过97，000个API查询。结果展示了一系列脆弱性：六个模型实现了96%至100%的攻击成功率（ASB），四个模型表现出有意义的抵抗力，ASB范围从42%至78%;在相同的架构上启用扩展推理将ASC从97%降低到42%。这些发现表明，不同供应商的安全对齐质量存在很大差异，模型规模无法预测对抗稳健性，并且思维模式提供了可部署的安全增强。总的来说，这项工作确定了，无论模型规模如何，当前的对齐技术仍然从根本上容易受到自适应多转弯攻击的影响，同时将刻意推理确定为一个有希望的防御方向。



## **19. Toward Reliable Machine Unlearning: Theory, Algorithms, and Evaluation**

迈向可靠的机器去学习：理论、算法和评估 cs.LG

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06993v1) [paper-pdf](https://arxiv.org/pdf/2512.06993v1)

**Authors**: Ali Ebrahimpour-Boroojeny

**Abstract**: We propose new methodologies for both unlearning random set of samples and class unlearning and show that they outperform existing methods. The main driver of our unlearning methods is the similarity of predictions to a retrained model on both the forget and remain samples. We introduce Adversarial Machine UNlearning (AMUN), which surpasses prior state-of-the-art methods for image classification based on SOTA MIA scores. AMUN lowers the model's confidence on forget samples by fine-tuning on their corresponding adversarial examples. Through theoretical analysis, we identify factors governing AMUN's performance, including smoothness. To facilitate training of smooth models with a controlled Lipschitz constant, we propose FastClip, a scalable method that performs layer-wise spectral-norm clipping of affine layers. In a separate study, we show that increased smoothness naturally improves adversarial example transfer, thereby supporting the second factor above.   Following the same principles for class unlearning, we show that existing methods fail in replicating a retrained model's behavior by introducing a nearest-neighbor membership inference attack (MIA-NN) that uses the probabilities assigned to neighboring classes to detect unlearned samples and demonstrate the vulnerability of such methods. We then propose a fine-tuning objective that mitigates this leakage by approximating, for forget-class inputs, the distribution over remaining classes that a model retrained from scratch would produce. To construct this approximation, we estimate inter-class similarity and tilt the target model's distribution accordingly. The resulting Tilted ReWeighting(TRW) distribution serves as the desired target during fine-tuning. Across multiple benchmarks, TRW matches or surpasses existing unlearning methods on prior metrics.

摘要: 我们提出了用于取消随机样本集和类别取消学习的新方法，并表明它们优于现有方法。我们取消学习方法的主要驱动因素是预测与忘记和保留样本的重新训练模型的相似性。我们引入了对抗机器非学习（AMUN），它超越了先前基于SOTA MIA评分的图像分类的最新技术。AMUN通过微调相应的对抗性示例来降低模型对忘记样本的信心。通过理论分析，我们确定了影响AMUN性能的因素，包括流畅性。为了促进具有受控Lipschitz常数的平滑模型的训练，我们提出了FastTrap，这是一种可扩展的方法，可以执行仿射层的逐层谱规范剪裁。在另一项研究中，我们表明平滑度的增加自然会改善对抗性示例转移，从而支持上述第二个因素。   遵循与类去学习相同的原则，我们表明现有方法无法通过引入最近邻成员资格推理攻击（MIA-NN）来复制重新训练的模型的行为，该攻击使用分配给邻近类的概率来检测未学习的样本并证明此类方法的脆弱性。然后，我们提出了一个微调目标，通过对忘记类输入逼近从头重新训练的模型在剩余类上产生的分布来减轻这种泄漏。为了构建这种逼近，我们估计类间相似性并相应地倾斜目标模型的分布。由此产生的倾斜重新加权（TRW）分布作为微调期间的所需目标。在多个基准测试中，TRW在先前指标上匹配或超越了现有的取消学习方法。



## **20. Patronus: Identifying and Mitigating Transferable Backdoors in Pre-trained Language Models**

守护者：识别和缓解预训练语言模型中的可转移后门 cs.CR

Work in progress

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06899v1) [paper-pdf](https://arxiv.org/pdf/2512.06899v1)

**Authors**: Tianhang Zhao, Wei Du, Haodong Zhao, Sufeng Duan, Gongshen Liu

**Abstract**: Transferable backdoors pose a severe threat to the Pre-trained Language Models (PLMs) supply chain, yet defensive research remains nascent, primarily relying on detecting anomalies in the output feature space. We identify a critical flaw that fine-tuning on downstream tasks inevitably modifies model parameters, shifting the output distribution and rendering pre-computed defense ineffective. To address this, we propose Patronus, a novel framework that use input-side invariance of triggers against parameter shifts. To overcome the convergence challenges of discrete text optimization, Patronus introduces a multi-trigger contrastive search algorithm that effectively bridges gradient-based optimization with contrastive learning objectives. Furthermore, we employ a dual-stage mitigation strategy combining real-time input monitoring with model purification via adversarial training. Extensive experiments across 15 PLMs and 10 tasks demonstrate that Patronus achieves $\geq98.7\%$ backdoor detection recall and reduce attack success rates to clean settings, significantly outperforming all state-of-the-art baselines in all settings. Code is available at https://github.com/zth855/Patronus.

摘要: 可转移后门对预训练语言模型（PLM）供应链构成严重威胁，但防御性研究仍处于起步阶段，主要依赖于检测输出特征空间中的异常。我们发现了一个关键缺陷，即对下游任务的微调不可避免地会修改模型参数，从而改变输出分布并使预先计算的防御无效。为了解决这个问题，我们提出了Patronus，这是一个新颖的框架，使用触发器的输入端不变性来对抗参数变化。为了克服离散文本优化的收敛挑战，Patronus引入了一种多触发对比搜索算法，该算法有效地将基于梯度的优化与对比学习目标联系起来。此外，我们采用了双阶段缓解策略，将实时输入监控与通过对抗训练的模型净化相结合。针对15个PLM和10个任务的广泛实验表明，Patronus实现了$$\geq98.7\%$后门检测召回，并降低了攻击成功率以清理设置，在所有设置中显着优于所有最先进的基线。代码可在https://github.com/zth855/Patronus上获取。



## **21. Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization**

三思而后行：本地化对抗模拟的理性代理框架 cs.CR

16 pages, 6 figures

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06713v1) [paper-pdf](https://arxiv.org/pdf/2512.06713v1)

**Authors**: Donghang Duan, Xu Zheng

**Abstract**: Current LLM-based text anonymization frameworks usually rely on remote API services from powerful LLMs, which creates an inherent "privacy paradox": users must somehow disclose data to untrusted third parties for superior privacy preservation. Moreover, directly migrating these frameworks to local small-scale models (LSMs) offers a suboptimal solution with catastrophic collapse in utility based on our core findings. Our work argues that this failure stems not merely from the capability deficits of LSMs, but from the inherent irrationality of the greedy adversarial strategies employed by current state-of-the-art (SoTA) methods. We model the anonymization process as a trade-off between Marginal Privacy Gain (MPG) and Marginal Utility Cost (MUC), and demonstrate that greedy strategies inevitably drift into an irrational state. To address this, we propose Rational Localized Adversarial Anonymization (RLAA), a fully localized and training-free framework featuring an Attacker-Arbitrator-Anonymizer (A-A-A) architecture. RLAA introduces an arbitrator that acts as a rationality gatekeeper, validating the attacker's inference to filter out feedback providing negligible benefits on privacy preservation. This mechanism enforces a rational early-stopping criterion, and systematically prevents utility collapse. Extensive experiments on different datasets demonstrate that RLAA achieves the best privacy-utility trade-off, and in some cases even outperforms SoTA on the Pareto principle. Our code and datasets will be released upon acceptance.

摘要: 当前基于LLM的文本匿名化框架通常依赖于强大的LLM的远程API服务，这造成了固有的“隐私悖论”：用户必须以某种方式向不受信任的第三方披露数据，以获得更好的隐私保护。此外，根据我们的核心发现，将这些框架直接迁移到本地小规模模型（LSM）提供了一个次优解决方案，但效用会灾难性崩溃。我们的工作认为，这种失败不仅源于LSM的能力缺陷，还源于当前最先进的（SoTA）方法所采用的贪婪对抗策略固有的不合理性。我们将匿名化过程建模为边缘隐私收益（MPG）和边缘公用事业成本（MUC）之间的权衡，并证明贪婪策略不可避免地陷入非理性状态。为了解决这个问题，我们提出了理性本地化对抗模拟（RLAA），这是一个完全本地化且免训练的框架，具有攻击者-模拟者-模拟者（A-A-A）架构。RLAA引入了一个仲裁员，充当理性守门人，验证攻击者的推断，以过滤掉对隐私保护提供微不足道好处的反馈。该机制强制执行理性的提前停止标准，并系统性地防止效用崩溃。对不同数据集的大量实验表明，RLAA实现了最佳的隐私与公用事业权衡，在某些情况下甚至在帕累托原则上优于SoTA。我们的代码和数据集将在接受后发布。



## **22. GSAE: Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering**

GAE：图形正规化稀疏自动编码器，用于稳健的LLM安全转向 cs.LG

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06655v1) [paper-pdf](https://arxiv.org/pdf/2512.06655v1)

**Authors**: Jehyeok Yeon, Federico Cinus, Yifan Wu, Luca Luceri

**Abstract**: Large language models (LLMs) face critical safety challenges, as they can be manipulated to generate harmful content through adversarial prompts and jailbreak attacks. Many defenses are typically either black-box guardrails that filter outputs, or internals-based methods that steer hidden activations by operationalizing safety as a single latent feature or dimension. While effective for simple concepts, this assumption is limiting, as recent evidence shows that abstract concepts such as refusal and temporality are distributed across multiple features rather than isolated in one. To address this limitation, we introduce Graph-Regularized Sparse Autoencoders (GSAEs), which extends SAEs with a Laplacian smoothness penalty on the neuron co-activation graph. Unlike standard SAEs that assign each concept to a single latent feature, GSAEs recover smooth, distributed safety representations as coherent patterns spanning multiple features. We empirically demonstrate that GSAE enables effective runtime safety steering, assembling features into a weighted set of safety-relevant directions and controlling them with a two-stage gating mechanism that activates interventions only when harmful prompts or continuations are detected during generation. This approach enforces refusals adaptively while preserving utility on benign queries. Across safety and QA benchmarks, GSAE steering achieves an average 82% selective refusal rate, substantially outperforming standard SAE steering (42%), while maintaining strong task accuracy (70% on TriviaQA, 65% on TruthfulQA, 74% on GSM8K). Robustness experiments further show generalization across LLaMA-3, Mistral, Qwen, and Phi families and resilience against jailbreak attacks (GCG, AutoDAN), consistently maintaining >= 90% refusal of harmful content.

摘要: 大型语言模型（LLM）面临着严峻的安全挑战，因为它们可能会被操纵以通过对抗性提示和越狱攻击生成有害内容。许多防御通常是过滤输出的黑匣子护栏，或者是基于内部的方法，通过将安全性作为单个潜在特征或维度来操作来引导隐藏激活。虽然对简单概念有效，但这种假设具有局限性，因为最近的证据表明，拒绝和时间性等抽象概念分布在多个特征中，而不是孤立在一个特征中。为了解决这一局限性，我们引入了图正规化稀疏自动编码器（GSAEs），它在神经元共激活图上使用拉普拉斯光滑罚分来扩展SAEs。与将每个概念分配给一个潜在特征的标准严重不良事件不同，G严重不良事件恢复为跨越多个特征的连贯模式的平滑、分布式的安全性表示。我们经验证明，GAE能够实现有效的运行时安全转向，将功能组装到一组加权的安全相关方向中，并通过两级门控机制控制它们，该机制仅在生成期间检测到有害提示或延续时激活干预。这种方法自适应地强制拒绝，同时保留对良性查询的实用性。在安全和QA基准中，GSE转向平均实现了82%的选择拒绝率，大大优于标准的SAS转向（42%），同时保持了强大的任务准确性（TriviaQA为70%，TruthfulQA为65%，GSM 8 K为74%）。鲁棒性实验进一步显示了LLaMA-3、Mistral、Qwen和Phi家族的普遍性以及对越狱攻击的韧性（GCG、AutoDAN），始终保持>= 90%拒绝有害内容。



## **23. Characterizing Large-Scale Adversarial Activities Through Large-Scale Honey-Nets**

通过大规模蜜网描述大规模对抗活动 cs.CR

Accepted at Conference IEEE UEMCON 2025

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06557v1) [paper-pdf](https://arxiv.org/pdf/2512.06557v1)

**Authors**: Tonia Haikal, Eman Hammad, Shereen Ismail

**Abstract**: The increasing sophistication of cyber threats demands novel approaches to characterize adversarial strategies, particularly those targeting critical infrastructure and IoT ecosystems. This paper presents a longitudinal analysis of attacker behavior using HoneyTrap, an adaptive honeypot framework deployed across geographically distributed nodes to emulate vulnerable services and safely capture malicious traffic. Over a 24 day observation window, more than 60.3 million events were collected. To enable scalable analytics, raw JSON logs were transformed into Apache Parquet, achieving 5.8 - 9.3x compression and 7.2x faster queries, while ASN enrichment and salted SHA-256 pseudonymization added network intelligence and privacy preservation.   Our analysis reveals three key findings: (1) The majority of traffic targeted HTTP and HTTPS services (ports 80 and 443), with more than 8 million connection attempts and daily peaks exceeding 1.7 million events. (2) SSH (port 22) was frequently subject to brute-force attacks, with over 4.6 million attempts. (3) Less common services like Minecraft (25565) and SMB (445) were also targeted, with Minecraft receiving about 118,000 daily attempts that often coincided with spikes on other ports.

摘要: 网络威胁日益复杂，需要采用新颖的方法来描述对抗策略，特别是针对关键基础设施和物联网生态系统的策略。本文使用HoneyTrap对攻击者行为进行了纵向分析，HoneyTrap是一种部署在地理分布的节点上的自适应蜜罐框架，用于模拟脆弱的服务并安全捕获恶意流量。在24天的观察窗口内，收集了超过6030万个事件。为了实现可扩展的分析，原始SON日志被转换为Apache Parquet，实现了5.8 - 9.3倍的压缩和7.2倍的查询速度，而收件箱浓缩和咸味SHA-256假名化增加了网络智能和隐私保护。   我们的分析揭示了三个关键发现：（1）大多数流量针对的是HTTP和HTTPS服务（端口80和443），连接尝试超过800万次，每日峰值超过170万次事件。(2)SSH（端口22）经常受到暴力攻击，尝试次数超过460万次。(3)Minecraft（25565）和SMB（445）等不太常见的服务也成为目标，Minecraft每天收到约118，000次尝试，通常与其他端口的高峰同时发生。



## **24. Securing the Model Context Protocol: Defending LLMs Against Tool Poisoning and Adversarial Attacks**

保护模型上下文协议：保护LLM免受工具中毒和对抗性攻击 cs.CR

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06556v1) [paper-pdf](https://arxiv.org/pdf/2512.06556v1)

**Authors**: Saeid Jamshidi, Kawser Wazed Nafi, Arghavan Moradi Dakhel, Negar Shahabi, Foutse Khomh, Naser Ezzati-Jivan

**Abstract**: The Model Context Protocol (MCP) enables Large Language Models to integrate external tools through structured descriptors, increasing autonomy in decision-making, task execution, and multi-agent workflows. However, this autonomy creates a largely overlooked security gap. Existing defenses focus on prompt-injection attacks and fail to address threats embedded in tool metadata, leaving MCP-based systems exposed to semantic manipulation. This work analyzes three classes of semantic attacks on MCP-integrated systems: (1) Tool Poisoning, where adversarial instructions are hidden in tool descriptors; (2) Shadowing, where trusted tools are indirectly compromised through contaminated shared context; and (3) Rug Pulls, where descriptors are altered after approval to subvert behavior. To counter these threats, we introduce a layered security framework with three components: RSA-based manifest signing to enforce descriptor integrity, LLM-on-LLM semantic vetting to detect suspicious tool definitions, and lightweight heuristic guardrails that block anomalous tool behavior at runtime. Through evaluation of GPT-4, DeepSeek, and Llama-3.5 across eight prompting strategies, we find that security performance varies widely by model architecture and reasoning method. GPT-4 blocks about 71 percent of unsafe tool calls, balancing latency and safety. DeepSeek shows the highest resilience to Shadowing attacks but with greater latency, while Llama-3.5 is fastest but least robust. Our results show that the proposed framework reduces unsafe tool invocation rates without model fine-tuning or internal modification.

摘要: 模型上下文协议（HCP）使大型语言模型能够通过结构化描述符集成外部工具，增加决策、任务执行和多代理工作流程的自主性。然而，这种自主性造成了一个很大程度上被忽视的安全漏洞。现有的防御措施专注于预算注入攻击，无法解决工具元数据中嵌入的威胁，从而使基于MVP的系统面临语义操纵。这项工作分析了对HCP集成系统的三类语义攻击：（1）工具中毒，其中对抗性指令隐藏在工具描述符中;（2）影子，其中受信任的工具通过受污染的共享上下文间接受到损害;（3）Rug Pull，其中描述符在批准后被更改以颠覆行为。为了应对这些威胁，我们引入了一个由三个组件组成的分层安全框架：基于RSA的清单签名以强制描述符完整性、LLM对LLM语义审查以检测可疑工具定义，以及在运行时阻止异常工具行为的轻量级启发式护栏。通过对八种提示策略的GPT-4、DeepSeek和Llama-3.5进行评估，我们发现安全性能因模型架构和推理方法而存在很大差异。GPT-4阻止约71%的不安全工具调用，平衡延迟和安全性。DeepSeek对Shadowing攻击表现出最高的弹性，但延迟更大，而Llama-3.5速度最快，但最不稳健。我们的结果表明，所提出的框架无需模型微调或内部修改即可降低不安全工具调用率。



## **25. Web Technologies Security in the AI Era: A Survey of CDN-Enhanced Defenses**

人工智能时代的Web技术安全：对CDO增强防御的调查 cs.CR

Accepted at 2025 IEEE Asia Pacific Conference on Wireless and Mobile (APWiMob). 7 pages, 5 figures

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06390v1) [paper-pdf](https://arxiv.org/pdf/2512.06390v1)

**Authors**: Mehrab Hosain, Sabbir Alom Shuvo, Matthew Ogbe, Md Shah Jalal Mazumder, Yead Rahman, Md Azizul Hakim, Anukul Pandey

**Abstract**: The modern web stack, which is dominated by browser-based applications and API-first backends, now operates under an adversarial equilibrium where automated, AI-assisted attacks evolve continuously. Content Delivery Networks (CDNs) and edge computing place programmable defenses closest to users and bots, making them natural enforcement points for machine-learning (ML) driven inspection, throttling, and isolation. This survey synthesizes the landscape of AI-enhanced defenses deployed at the edge: (i) anomaly- and behavior-based Web Application Firewalls (WAFs) within broader Web Application and API Protection (WAAP), (ii) adaptive DDoS detection and mitigation, (iii) bot management that resists human-mimicry, and (iv) API discovery, positive security modeling, and encrypted-traffic anomaly analysis. We add a systematic survey method, a threat taxonomy mapped to edge-observable signals, evaluation metrics, deployment playbooks, and governance guidance. We conclude with a research agenda spanning XAI, adversarial robustness, and autonomous multi-agent defense. Our findings indicate that edge-centric AI measurably improves time-to-detect and time-to-mitigate while reducing data movement and enhancing compliance, yet introduces new risks around model abuse, poisoning, and governance.

摘要: 现代Web栈由基于浏览器的应用程序和API优先的后台主导，现在在敌对平衡下运行，自动化的人工智能辅助攻击不断发展。内容交付网络（CDO）和边缘计算将可编程防御设置在最接近用户和机器人的地方，使其成为机器学习（ML）驱动的检查、限制和隔离的自然执行点。这项调查综合了在边缘部署的人工智能增强防御的格局：（i）更广泛的Web应用程序和API保护（WAAP）中的基于异常和行为的Web应用防火墙（WAF），（ii）自适应分布式拒绝服务检测和缓解，（iii）抵抗人类模仿的机器人管理，以及（iv）API发现、积极安全建模和预测流量异常分析。我们添加了系统性调查方法、映射到边缘可观察信号的威胁分类、评估指标、部署剧本和治理指南。我们最后提出了一个涵盖XAI、对抗鲁棒性和自主多智能体防御的研究议程。我们的研究结果表明，以边缘为中心的人工智能可以显着地提高检测时间和缓解时间，同时减少数据移动并提高合规性，但也带来了围绕模型滥用、中毒和治理的新风险。



## **26. Degrading Voice: A Comprehensive Overview of Robust Voice Conversion Through Input Manipulation**

有辱人格的声音：通过输入操纵实现稳健语音转换的全面概述 eess.AS

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06304v1) [paper-pdf](https://arxiv.org/pdf/2512.06304v1)

**Authors**: Xining Song, Zhihua Wei, Rui Wang, Haixiao Hu, Yanxiang Chen, Meng Han

**Abstract**: Identity, accent, style, and emotions are essential components of human speech. Voice conversion (VC) techniques process the speech signals of two input speakers and other modalities of auxiliary information such as prompts and emotion tags. It changes para-linguistic features from one to another, while maintaining linguistic contents. Recently, VC models have made rapid advancements in both generation quality and personalization capabilities. These developments have attracted considerable attention for diverse applications, including privacy preservation, voice-print reproduction for the deceased, and dysarthric speech recovery. However, these models only learn non-robust features due to the clean training data. Subsequently, it results in unsatisfactory performances when dealing with degraded input speech in real-world scenarios, including additional noise, reverberation, adversarial attacks, or even minor perturbation. Hence, it demands robust deployments, especially in real-world settings. Although latest researches attempt to find potential attacks and countermeasures for VC systems, there remains a significant gap in the comprehensive understanding of how robust the VC model is under input manipulation. here also raises many questions: For instance, to what extent do different forms of input degradation attacks alter the expected output of VC models? Is there potential for optimizing these attack and defense strategies? To answer these questions, we classify existing attack and defense methods from the perspective of input manipulation and evaluate the impact of degraded input speech across four dimensions, including intelligibility, naturalness, timbre similarity, and subjective perception. Finally, we outline open issues and future directions.

摘要: 身份、口音、风格和情感是人类言语的重要组成部分。语音转换（VC）技术处理两个输入扬声器的语音信号以及其他辅助信息形式，例如提示和情感标签。它将准语言特征从一个转变到另一个，同时保持语言内容。近年来，风险投资模式在世代质量和个性化能力方面都取得了快速进步。这些发展在各种应用中引起了相当大的关注，包括隐私保护、死者声纹复制和构音障碍言语恢复。然而，由于训练数据干净，这些模型仅学习非稳健特征。随后，在现实世界场景中处理降级的输入语音时，包括额外的噪音、回响、对抗性攻击，甚至轻微的干扰，会导致性能不理想。因此，它需要强大的部署，尤其是在现实世界环境中。尽管最新的研究试图寻找针对VC系统的潜在攻击和对策，但在全面理解VC模型在输入操纵下的稳健性方面仍然存在显着差距。这里也提出了许多问题：例如，不同形式的输入降级攻击在多大程度上改变了风险投资模型的预期输出？是否有潜力优化这些攻击和防御策略？为了回答这些问题，我们从输入操纵的角度对现有的攻击和防御方法进行了分类，并评估降级输入语音在四个维度上的影响，包括可懂度、自然度、音色相似性和主观感知。最后，我们概述了悬而未决的问题和未来的方向。



## **27. Sparse Neural Approximations for Bilevel Adversarial Problems in Power Grids**

电网二层对抗问题的稀疏神经逼近 eess.SY

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06187v1) [paper-pdf](https://arxiv.org/pdf/2512.06187v1)

**Authors**: Young-ho Cho, Harsha Nagarajan, Deepjyoti Deka, Hao Zhu

**Abstract**: The adversarial worst-case load shedding (AWLS) problem is pivotal for identifying critical contingencies under line outages. It is naturally cast as a bilevel program: the upper level simulates an attacker determining worst-case line failures, and the lower level corresponds to the defender's generator redispatch operations. Conventional techniques using optimality conditions render the bilevel, mixed-integer formulation computationally prohibitive due to the combinatorial number of topologies and the nonconvexity of AC power flow constraints. To address these challenges, we develop a novel single-level optimal value-function (OVF) reformulation and further leverage a data-driven neural network (NN) surrogate of the follower's optimal value. To ensure physical realizability, we embed the trained surrogate in a physics-constrained NN (PCNN) formulation that couples the OVF inequality with (relaxed) AC feasibility, yielding a mixed-integer convex model amenable to off-the-shelf solvers. To achieve scalability, we learn a sparse, area-partitioned NN via spectral clustering; the resulting block-sparse architecture scales essentially linearly with system size while preserving accuracy. Notably, our approach produces near-optimal worst-case failures and generalizes across loading conditions and unseen topologies, enabling rapid online recomputation. Numerical experiments on the IEEE 14- and 118-bus systems demonstrate the method's scalability and solution quality for large-scale contingency analysis, with an average optimality gap of 5.8% compared to conventional methods, while maintaining computation times under one minute.

摘要: 对抗性的最坏情况甩负荷（AWLS）问题对于识别线路停电情况下的关键意外情况至关重要。它自然被视为一个二层程序：上层模拟攻击者确定最坏情况的线路故障，下层对应于防御者的生成器重新调度操作。由于布局的组合数量和AC潮流约束的非凸性，使用最优性条件的传统技术使得二层混合整公式在计算上难以接受。为了应对这些挑战，我们开发了一种新型的单级最优价值函数（OVF）重新公式，并进一步利用跟随者最优价值的数据驱动神经网络（NN）替代品。为了确保物理可实现性，我们将训练好的代理嵌入物理约束神经网络（PCNN）公式中，该公式将OVF不等式与（宽松的）AC可行性相结合，产生适合现成求解器的混合整凸模型。为了实现可扩展性，我们通过谱集群学习稀疏、区域分区的神经网络;由此产生的块稀疏架构基本上随系统大小线性扩展，同时保持准确性。值得注意的是，我们的方法会产生近乎最优的最坏情况故障，并在负载条件和未见的布局中进行推广，从而实现快速在线重新计算。在IEEE 14和118节点系统上进行的数值实验证明了该方法对于大规模意外分析的可扩展性和解决方案质量，与传统方法相比，平均最优性差距为5.8%，同时将计算时间保持在1分钟以下。



## **28. DEFEND: Poisoned Model Detection and Malicious Client Exclusion Mechanism for Secure Federated Learning-based Road Condition Classification**

DEFEND：用于安全的基于联邦学习的道路状况分类的中毒模型检测和恶意客户端排除机制 cs.CR

Accepted to the 41st ACM/SIGAPP Symposium on Applied Computing (SAC 2026)

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06172v1) [paper-pdf](https://arxiv.org/pdf/2512.06172v1)

**Authors**: Sheng Liu, Panos Papadimitratos

**Abstract**: Federated Learning (FL) has drawn the attention of the Intelligent Transportation Systems (ITS) community. FL can train various models for ITS tasks, notably camera-based Road Condition Classification (RCC), in a privacy-preserving collaborative way. However, opening up to collaboration also opens FL-based RCC systems to adversaries, i.e., misbehaving participants that can launch Targeted Label-Flipping Attacks (TLFAs) and threaten transportation safety. Adversaries mounting TLFAs poison training data to misguide model predictions, from an actual source class (e.g., wet road) to a wrongly perceived target class (e.g., dry road). Existing countermeasures against poisoning attacks cannot maintain model performance under TLFAs close to the performance level in attack-free scenarios, because they lack specific model misbehavior detection for TLFAs and neglect client exclusion after the detection. To close this research gap, we propose DEFEND, which includes a poisoned model detection strategy that leverages neuron-wise magnitude analysis for attack goal identification and Gaussian Mixture Model (GMM)-based clustering. DEFEND discards poisoned model contributions in each round and adapts accordingly client ratings, eventually excluding malicious clients. Extensive evaluation involving various FL-RCC models and tasks shows that DEFEND can thwart TLFAs and outperform seven baseline countermeasures, with at least 15.78% improvement, with DEFEND remarkably achieving under attack the same performance as in attack-free scenarios.

摘要: 联邦学习（FL）引起了智能交通系统（ITS）社区的关注。FL可以以隐私保护的协作方式为ITS任务训练各种模型，特别是基于摄像头的道路状况分类（RCC）。然而，开放合作也会向对手开放基于FL的RCC系统，即，行为不端的参与者可以发起有针对性的标签翻转攻击（TLFA）并威胁运输安全。安装TLR的对手会毒害训练数据，以误导来自实际源类的模型预测（例如，潮湿的道路）到错误感知的目标类别（例如，干燥的道路）。现有的针对中毒攻击的对策无法将TLR下的模型性能保持在无攻击场景中的性能水平附近，因为它们缺乏针对TLR的特定模型不当行为检测，并且忽视了检测后的客户端排除。为了缩小这一研究差距，我们提出了DEFEND，其中包括一种中毒模型检测策略，该策略利用神经元幅度分析来识别攻击目标和基于高斯混合模型（GMM）的集群。DEFEND会在每一轮中丢弃有毒的模型贡献，并相应地调整客户评级，最终排除恶意客户。涉及各种FL-CC模型和任务的广泛评估表明，DEFEND可以阻止TLR并优于七种基线对策，改进至少15.78%，其中DEFEND在攻击下显着实现了与无攻击场景相同的性能。



## **29. Toward Patch Robustness Certification and Detection for Deep Learning Systems Beyond Consistent Samples**

超越一致样本的深度学习系统的补丁稳健性认证和检测 cs.SE

accepted by IEEE Transactions on Reliability; extended technical report

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06123v1) [paper-pdf](https://arxiv.org/pdf/2512.06123v1)

**Authors**: Qilin Zhou, Zhengyuan Wei, Haipeng Wang, Zhuo Wang, W. K. Chan

**Abstract**: Patch robustness certification is an emerging kind of provable defense technique against adversarial patch attacks for deep learning systems. Certified detection ensures the detection of all patched harmful versions of certified samples, which mitigates the failures of empirical defense techniques that could (easily) be compromised. However, existing certified detection methods are ineffective in certifying samples that are misclassified or whose mutants are inconsistently pre icted to different labels. This paper proposes HiCert, a novel masking-based certified detection technique. By focusing on the problem of mutants predicted with a label different from the true label with our formal analysis, HiCert formulates a novel formal relation between harmful samples generated by identified loopholes and their benign counterparts. By checking the bound of the maximum confidence among these potentially harmful (i.e., inconsistent) mutants of each benign sample, HiCert ensures that each harmful sample either has the minimum confidence among mutants that are predicted the same as the harmful sample itself below this bound, or has at least one mutant predicted with a label different from the harmful sample itself, formulated after two novel insights. As such, HiCert systematically certifies those inconsistent samples and consistent samples to a large extent. To our knowledge, HiCert is the first work capable of providing such a comprehensive patch robustness certification for certified detection. Our experiments show the high effectiveness of HiCert with a new state-of the-art performance: It certifies significantly more benign samples, including those inconsistent and consistent, and achieves significantly higher accuracy on those samples without warnings and a significantly lower false silent ratio.

摘要: 补丁稳健性认证是一种新兴的可证明防御技术，针对深度学习系统的对抗性补丁攻击。认证检测可确保检测到认证样本的所有修补有害版本，从而减轻了可能（容易）受到损害的经验防御技术的失败。然而，现有的经认证的检测方法在认证被错误分类或其突变体被不一致地预测到不同标签的样本方面无效。本文提出了HiCert，这是一种新型的基于掩蔽的认证检测技术。通过我们的正式分析关注使用与真实标签不同的标签预测的突变体问题，HiCert在已识别漏洞产生的有害样本与其良性样本之间建立了一种新颖的形式关系。通过检查这些潜在有害因素（即，每个良性样本的突变体不一致），HiCert确保每个有害样本要么在预测与有害样本本身相同的突变体中具有最低置信度，低于此界限，要么至少有一个突变体预测，其标签与有害样本本身不同，根据两个新的见解制定。因此，HiCert系统地认证那些不一致的样本和很大程度上一致的样本。据我们所知，HiCert是第一款能够为认证检测提供如此全面的补丁稳健性认证的作品。我们的实验表明，HiCert具有高效率，具有最新的最先进性能：它认证了明显更良性的样本，包括那些不一致和一致的样本，并在没有警告的情况下实现了显着更高的准确性和显着更低的假沉默率。



## **30. When Privacy Isn't Synthetic: Hidden Data Leakage in Generative AI Models**

当隐私不是合成的：生成人工智能模型中的隐藏数据泄露 cs.LG

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06062v1) [paper-pdf](https://arxiv.org/pdf/2512.06062v1)

**Authors**: S. M. Mustaqim, Anantaa Kotal, Paul H. Yi

**Abstract**: Generative models are increasingly used to produce privacy-preserving synthetic data as a safe alternative to sharing sensitive training datasets. However, we demonstrate that such synthetic releases can still leak information about the underlying training samples through structural overlap in the data manifold. We propose a black-box membership inference attack that exploits this vulnerability without requiring access to model internals or real data. The attacker repeatedly queries the generative model to obtain large numbers of synthetic samples, performs unsupervised clustering to identify dense regions of the synthetic distribution, and then analyzes cluster medoids and neighborhoods that correspond to high-density regions in the original training data. These neighborhoods act as proxies for training samples, enabling the adversary to infer membership or reconstruct approximate records. Our experiments across healthcare, finance, and other sensitive domains show that cluster overlap between real and synthetic data leads to measurable membership leakage-even when the generator is trained with differential privacy or other noise mechanisms. The results highlight an under-explored attack surface in synthetic data generation pipelines and call for stronger privacy guarantees that account for distributional neighborhood inference rather than sample-level memorization alone, underscoring its role in privacy-preserving data publishing. Implementation and evaluation code are publicly available at:github.com/Cluster-Medoid-Leakage-Attack.

摘要: 生成模型越来越多地用于生成保护隐私的合成数据，作为共享敏感训练数据集的安全替代方案。然而，我们证明，此类合成发布仍然可以通过数据集合中的结构重叠泄露有关基础训练样本的信息。我们提出了一种黑匣子成员资格推断攻击，可以利用此漏洞，而无需访问模型内部或真实数据。攻击者反复查询生成模型以获取大量合成样本，执行无监督集群以识别合成分布的密集区域，然后分析与原始训练数据中高密度区域对应的集群中间段和邻居。这些邻居充当训练样本的代理，使对手能够推断成员资格或重建大致记录。我们在医疗保健、金融和其他敏感领域的实验表明，真实数据和合成数据之间的集群重叠会导致可测量的成员泄露--即使生成器是经过差异隐私或其他噪音机制训练的。结果凸显了合成数据生成管道中未充分探索的攻击表面，并呼吁更强的隐私保证，以考虑分布式邻居推断，而不仅仅是样本级记忆，强调了其在隐私保护数据发布中的作用。实施和评估代码可在github.com/Cluster-Medoid-Leakage-Attack上公开获取。



## **31. Exposing Pink Slime Journalism: Linguistic Signatures and Robust Detection Against LLM-Generated Threats**

揭露Pink Slime新闻：语言签名和针对LLM生成的威胁的稳健检测 cs.CL

Published in RANLP 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05331v1) [paper-pdf](https://arxiv.org/pdf/2512.05331v1)

**Authors**: Sadat Shahriar, Navid Ayoobi, Arjun Mukherjee, Mostafa Musharrat, Sai Vishnu Vamsi

**Abstract**: The local news landscape, a vital source of reliable information for 28 million Americans, faces a growing threat from Pink Slime Journalism, a low-quality, auto-generated articles that mimic legitimate local reporting. Detecting these deceptive articles requires a fine-grained analysis of their linguistic, stylistic, and lexical characteristics. In this work, we conduct a comprehensive study to uncover the distinguishing patterns of Pink Slime content and propose detection strategies based on these insights. Beyond traditional generation methods, we highlight a new adversarial vector: modifications through large language models (LLMs). Our findings reveal that even consumer-accessible LLMs can significantly undermine existing detection systems, reducing their performance by up to 40% in F1-score. To counter this threat, we introduce a robust learning framework specifically designed to resist LLM-based adversarial attacks and adapt to the evolving landscape of automated pink slime journalism, and showed and improvement by up to 27%.

摘要: 当地新闻格局是2800万美国人可靠信息的重要来源，但面临着来自Pink Slime Journalism的日益增长的威胁，Pink Slime Journalism是一种模仿合法当地报道的低质量自动生成文章。检测这些欺骗性文章需要对其语言、文体和词汇特征进行细粒度分析。在这项工作中，我们进行了一项全面的研究，以揭示Pink Slime内容的区别模式，并根据这些见解提出检测策略。除了传统的生成方法之外，我们还强调了一种新的对抗性载体：通过大型语言模型（LLM）进行修改。我们的研究结果表明，即使是消费者可访问的LLM也会显着破坏现有的检测系统，使其F1评分的性能降低高达40%。为了应对这一威胁，我们引入了一个强大的学习框架，专门设计用于抵抗基于LLM的对抗攻击并适应自动化粉红粘液新闻不断变化的格局，并表现出高达27%的改进。



## **32. VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language**

VEIL：通过隐性语言的视觉开发破解文本到视频模型 cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.13127v2) [paper-pdf](https://arxiv.org/pdf/2511.13127v2)

**Authors**: Zonghao Ying, Moyang Chen, Nizhang Li, Zhiqiang Wang, Wenxin Zhang, Quanchen Zou, Zonglei Jing, Aishan Liu, Xianglong Liu

**Abstract**: Jailbreak attacks can circumvent model safety guardrails and reveal critical blind spots. Prior attacks on text-to-video (T2V) models typically add adversarial perturbations to obviously unsafe prompts, which are often easy to detect and defend. In contrast, we show that benign-looking prompts containing rich, implicit cues can induce T2V models to generate semantically unsafe videos that both violate policy and preserve the original (blocked) intent. To realize this, we propose VEIL, a jailbreak framework that leverages T2V models' cross-modal associative patterns via a modular prompt design. Specifically, our prompts combine three components: neutral scene anchors, which provide the surface-level scene description extracted from the blocked intent to maintain plausibility; latent auditory triggers, textual descriptions of innocuous-sounding audio events (e.g., creaking, muffled noises) that exploit learned audio-visual co-occurrence priors to bias the model toward particular unsafe visual concepts; and stylistic modulators, cinematic directives (e.g., camera framing, atmosphere) that amplify and stabilize the latent trigger's effect. We formalize attack generation as a constrained optimization over the above modular prompt space and solve it with a guided search procedure that balances stealth and effectiveness. Extensive experiments over 7 T2V models demonstrate the efficacy of our attack, achieving a 23 percent improvement in average attack success rate in commercial models. Our demos and codes can be found at https://github.com/NY1024/VEIL.

摘要: 越狱攻击可以绕过模型安全护栏并暴露关键盲点。先前对文本转视频（T2 V）模型的攻击通常会向明显不安全的提示添加对抗性扰动，而这些提示通常很容易检测和防御。相比之下，我们表明，包含丰富、隐性线索的看似友善的提示可以诱导T2 V模型生成语义不安全的视频，这些视频既违反了政策，又保留了原始（被阻止的）意图。为了实现这一点，我们提出了VEIL，这是一个越狱框架，通过模块化提示设计利用T2 V模型的跨模式关联模式。具体来说，我们的提示结合了三个组件：中性场景锚点，它提供从被阻止的意图中提取的表面级场景描述，以保持可信性;潜在听觉触发器，听起来无害的音频事件的文本描述（例如，吱吱作响、低沉的噪音），利用习得的视听同现先验来将模型偏向特定不安全的视觉概念;以及风格调节器、电影指令（例如，相机取景、大气）放大和稳定潜在触发的效果。我们将攻击生成形式化为对上述模块提示空间的约束优化，并通过平衡隐形性和有效性的引导搜索过程来解决它。对7个T2 V模型进行的广泛实验证明了我们攻击的有效性，使商业模型中的平均攻击成功率提高了23%。我们的演示和代码可在https://github.com/NY1024/VEIL上找到。



## **33. 3D-ANC: Adaptive Neural Collapse for Robust 3D Point Cloud Recognition**

3D-非国大：用于稳健3D点云识别的自适应神经崩溃 cs.CV

AAAI 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.07040v2) [paper-pdf](https://arxiv.org/pdf/2511.07040v2)

**Authors**: Yuanmin Huang, Wenxuan Li, Mi Zhang, Xiaohan Zhang, Xiaoyu You, Min Yang

**Abstract**: Deep neural networks have recently achieved notable progress in 3D point cloud recognition, yet their vulnerability to adversarial perturbations poses critical security challenges in practical deployments. Conventional defense mechanisms struggle to address the evolving landscape of multifaceted attack patterns. Through systematic analysis of existing defenses, we identify that their unsatisfactory performance primarily originates from an entangled feature space, where adversarial attacks can be performed easily. To this end, we present 3D-ANC, a novel approach that capitalizes on the Neural Collapse (NC) mechanism to orchestrate discriminative feature learning. In particular, NC depicts where last-layer features and classifier weights jointly evolve into a simplex equiangular tight frame (ETF) arrangement, establishing maximally separable class prototypes. However, leveraging this advantage in 3D recognition confronts two substantial challenges: (1) prevalent class imbalance in point cloud datasets, and (2) complex geometric similarities between object categories. To tackle these obstacles, our solution combines an ETF-aligned classification module with an adaptive training framework consisting of representation-balanced learning (RBL) and dynamic feature direction loss (FDL). 3D-ANC seamlessly empowers existing models to develop disentangled feature spaces despite the complexity in 3D data distribution. Comprehensive evaluations state that 3D-ANC significantly improves the robustness of models with various structures on two datasets. For instance, DGCNN's classification accuracy is elevated from 27.2% to 80.9% on ModelNet40 -- a 53.7% absolute gain that surpasses leading baselines by 34.0%.

摘要: 深度神经网络最近在3D点云识别方面取得了显着进展，但它们对对抗扰动的脆弱性在实际部署中构成了关键的安全挑战。传统的防御机制难以应对不断变化的多方面攻击模式。通过对现有防御系统的系统分析，我们发现它们不令人满意的性能主要源于纠缠的特征空间，其中可以很容易地进行对抗性攻击。为此，我们提出了3D-非国大，这是一种利用神经崩溃（NC）机制来协调区分性特征学习的新型方法。特别是，NC描述了最后一层特征和分类器权重联合演变为单形等距紧框架（ETF）排列的地方，从而建立最大可分离的类原型。然而，在3D识别中利用这一优势面临两个重大挑战：（1）点云数据集中普遍存在的类别不平衡，以及（2）对象类别之间复杂的几何相似性。为了解决这些障碍，我们的解决方案将ETF对齐的分类模块与由表示平衡学习（RBL）和动态特征方向丢失（FDL）组成的自适应训练框架相结合。3D-ANC无缝地使现有模型能够开发解纠缠的特征空间，尽管3D数据分布很复杂。综合评估表明，3D-ANC显着提高了两个数据集上具有不同结构的模型的稳健性。例如，DGCNN在ModelNet 40上的分类准确率从27.2%提高到80.9%，绝对增益为53.7%，超过领先基线34.0%。



## **34. ZQBA: Zero Query Black-box Adversarial Attack**

ZQBA：零查询黑匣子对抗攻击 cs.CV

Accepted in ICAART 2026 Conference

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2510.00769v2) [paper-pdf](https://arxiv.org/pdf/2510.00769v2)

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Current black-box adversarial attacks either require multiple queries or diffusion models to produce adversarial samples that can impair the target model performance. However, these methods require training a surrogate loss or diffusion models to produce adversarial samples, which limits their applicability in real-world settings. Thus, we propose a Zero Query Black-box Adversarial (ZQBA) attack that exploits the representations of Deep Neural Networks (DNNs) to fool other networks. Instead of requiring thousands of queries to produce deceiving adversarial samples, we use the feature maps obtained from a DNN and add them to clean images to impair the classification of a target model. The results suggest that ZQBA can transfer the adversarial samples to different models and across various datasets, namely CIFAR and Tiny ImageNet. The experiments also show that ZQBA is more effective than state-of-the-art black-box attacks with a single query, while maintaining the imperceptibility of perturbations, evaluated both quantitatively (SSIM) and qualitatively, emphasizing the vulnerabilities of employing DNNs in real-world contexts. All the source code is available at https://github.com/Joana-Cabral/ZQBA.

摘要: 当前的黑匣子对抗攻击要么需要多个查询，要么需要扩散模型来产生可能损害目标模型性能的对抗样本。然而，这些方法需要训练替代损失或扩散模型来产生对抗样本，这限制了它们在现实世界环境中的适用性。因此，我们提出了零查询黑匣子对抗（ZQBA）攻击，利用深度神经网络（DNN）的表示来愚弄其他网络。我们不需要数千个查询来产生具有欺骗性的对抗样本，而是使用从DNN获得的特征地图并将它们添加到干净的图像中，以损害目标模型的分类。结果表明，ZQBA可以将对抗样本传输到不同的模型和各种数据集，即CIFAR和Tiny ImageNet。实验还表明，ZQBA比单个查询的最先进的黑匣子攻击更有效，同时保持了扰动的不可感知性，并进行了定量（SSIM）和定性评估，强调了在现实世界中使用DNN的漏洞。所有源代码均可在https://github.com/Joana-Cabral/ZQBA上获取。



## **35. Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors**

Eyes-on-Me：通过可转移注意力引导吸引器的可扩展RAG中毒 cs.LG

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2510.00586v2) [paper-pdf](https://arxiv.org/pdf/2510.00586v2)

**Authors**: Yen-Shan Chen, Sian-Yao Huang, Cheng-Lin Yang, Yun-Nung Chen

**Abstract**: Existing data poisoning attacks on retrieval-augmented generation (RAG) systems scale poorly because they require costly optimization of poisoned documents for each target phrase. We introduce Eyes-on-Me, a modular attack that decomposes an adversarial document into reusable Attention Attractors and Focus Regions. Attractors are optimized to direct attention to the Focus Region. Attackers can then insert semantic baits for the retriever or malicious instructions for the generator, adapting to new targets at near zero cost. This is achieved by steering a small subset of attention heads that we empirically identify as strongly correlated with attack success. Across 18 end-to-end RAG settings (3 datasets $\times$ 2 retrievers $\times$ 3 generators), Eyes-on-Me raises average attack success rates from 21.9 to 57.8 (+35.9 points, 2.6$\times$ over prior work). A single optimized attractor transfers to unseen black box retrievers and generators without retraining. Our findings establish a scalable paradigm for RAG data poisoning and show that modular, reusable components pose a practical threat to modern AI systems. They also reveal a strong link between attention concentration and model outputs, informing interpretability research.

摘要: 对检索增强生成（RAG）系统的现有数据中毒攻击规模很差，因为它们需要对每个目标短语的中毒文档进行昂贵的优化。我们引入了Eyes-on-Me，这是一种模块化攻击，可以将对抗性文档分解为可重复使用的注意力吸引器和焦点区域。吸引器经过优化，以将注意力引导到焦点区域。然后，攻击者可以为检索器插入语义诱饵或为生成器插入恶意指令，以接近零的成本适应新目标。这是通过引导一小部分注意力头来实现的，我们根据经验认为这些注意力头与攻击成功密切相关。在18个端到端RAG设置中（3个数据集$\times $2 retroevers $\times $3个生成器），Eyes-on-Me将平均攻击成功率从21.9提高到57.8（+35.9分，2.6 $\times $比之前的工作）。单个优化的吸引子无需重新训练即可转移到看不见的黑匣子取回器和生成器。我们的研究结果为RAG数据中毒建立了一个可扩展的范式，并表明模块化、可重复使用的组件对现代人工智能系统构成了实际威胁。它们还揭示了注意力集中与模型输出之间的密切联系，为可解释性研究提供了信息。



## **36. Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy**

基于可解释自适应差分隐私的隐私保护分散联邦学习 cs.CR

20 pages

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2509.10691v2) [paper-pdf](https://arxiv.org/pdf/2509.10691v2)

**Authors**: Fardin Jalil Piran, Zhiling Chen, Yang Zhang, Qianyu Zhou, Jiong Tang, Farhad Imani

**Abstract**: Decentralized Federated Learning (DFL) enables collaborative model training without a central server, but it remains vulnerable to privacy leakage because shared model updates can expose sensitive information through inversion, reconstruction, and membership inference attacks. Differential Privacy (DP) provides formal safeguards, yet existing DP-enabled DFL methods operate as black-boxes that cannot track cumulative noise added across clients and rounds, forcing each participant to inject worst-case perturbations that severely degrade accuracy. We propose PrivateDFL, a new explainable and privacy-preserving framework that addresses this gap by combining a HyperDimensional Computing (HD) model with a transparent DP noise accountant tailored to decentralized learning. HD offers structured, noise-tolerant high-dimensional representations, while the accountant explicitly tracks cumulative perturbations so each client adds only the minimal incremental noise required to satisfy its (epsilon, delta) budget. This yields significantly tighter and more interpretable privacy-utility tradeoffs than prior DP-DFL approaches. Experiments on MNIST (image), ISOLET (speech), and UCI-HAR (wearable sensor) show that PrivateDFL consistently surpasses centralized DP-SGD and Renyi-DP Transformer and deep learning baselines under both IID and non-IID partitions, improving accuracy by up to 24.4% on MNIST, over 80% on ISOLET, and 14.7% on UCI-HAR, while reducing inference latency by up to 76 times and energy consumption by up to 36 times. These results position PrivateDFL as an efficient and trustworthy solution for privacy-sensitive pattern recognition applications such as healthcare, finance, human-activity monitoring, and industrial sensing. Future work will extend the accountant to adversarial participation, heterogeneous privacy budgets, and dynamic topologies.

摘要: 去中心化联邦学习（DFL）可以在没有中央服务器的情况下进行协作模型训练，但它仍然容易受到隐私泄露的影响，因为共享模型更新可能会通过倒置、重建和成员资格推断攻击暴露敏感信息。差异隐私（DP）提供了正式的保护措施，但现有的支持DP的DFL方法作为黑匣子运行，无法跟踪跨客户端和回合添加的累积噪音，迫使每个参与者注入最坏情况的干扰，从而严重降低准确性。我们提出PrivateDFL，这是一个新的可解释且保护隐私的框架，通过将超维计算（HD）模型与专为去中心学习量身定制的透明DP噪音会计器相结合来解决这一差距。HD提供结构化、耐噪的多维表示，而会计师明确跟踪累积扰动，因此每个客户仅添加满足其（NPS、Delta）预算所需的最小增量噪音。与之前的DP-DFL方法相比，这产生了明显更严格、更可解释的隐私与公用事业权衡。MNIST（图像）、ISOLET（语音）和UCI-HAR实验（可穿戴传感器）表明，PrivateDFL在IID和非IID分区下始终超过集中式DP-BCD和Renyi-DP Transformer以及深度学习基线，在MNIST上提高了高达24.4%的准确性，在ISOLET上提高了80%以上，在UCI-HAR上提高了14.7%，同时将推理延迟减少高达76倍，将能源消耗减少高达36倍。这些结果使PrivateDFL成为医疗保健、金融、人类活动监测和工业传感等隐私敏感模式识别应用的高效且值得信赖的解决方案。未来的工作将将会计扩展到对抗性参与、异类隐私预算和动态布局。



## **37. Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking**

你的还是我的？针对神经音频水印的覆盖攻击 cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.05835v2) [paper-pdf](https://arxiv.org/pdf/2509.05835v2)

**Authors**: Lingfeng Yao, Chenpei Huang, Shengyao Wang, Junpei Xue, Hanqing Guo, Jiang Liu, Phone Lin, Tomoaki Ohtsuki, Miao Pan

**Abstract**: As generative audio models are rapidly evolving, AI-generated audios increasingly raise concerns about copyright infringement and misinformation spread. Audio watermarking, as a proactive defense, can embed secret messages into audio for copyright protection and source verification. However, current neural audio watermarking methods focus primarily on the imperceptibility and robustness of watermarking, while ignoring its vulnerability to security attacks. In this paper, we develop a simple yet powerful attack: the overwriting attack that overwrites the legitimate audio watermark with a forged one and makes the original legitimate watermark undetectable. Based on the audio watermarking information that the adversary has, we propose three categories of overwriting attacks, i.e., white-box, gray-box, and black-box attacks. We also thoroughly evaluate the proposed attacks on state-of-the-art neural audio watermarking methods. Experimental results demonstrate that the proposed overwriting attacks can effectively compromise existing watermarking schemes across various settings and achieve a nearly 100% attack success rate. The practicality and effectiveness of the proposed overwriting attacks expose security flaws in existing neural audio watermarking systems, underscoring the need to enhance security in future audio watermarking designs.

摘要: 随着生成音频模型的迅速发展，人工智能生成的音频越来越引发人们对版权侵权和错误信息传播的担忧。音频水印作为一种主动防御，可以将秘密消息嵌入音频中，以实现版权保护和源验证。然而，目前的神经音频水印方法主要关注水印的不可感知性和鲁棒性，而忽视了其对安全攻击的脆弱性。在本文中，我们开发了一种简单但强大的攻击：MIDI攻击，用伪造的水印覆盖合法的音频水印，并使原始的合法水印无法检测。根据对手拥有的音频水印信息，我们提出了三种类型的MIDI攻击，即，白盒、灰盒和黑匣子攻击。我们还彻底评估了对最先进的神经音频水印方法提出的攻击。实验结果表明，提出的MIDI攻击可以有效地破坏各种设置中的现有水印方案，并达到近100%的攻击成功率。提出的MIDI攻击的实用性和有效性暴露了现有神经音频水印系统中的安全缺陷，强调了在未来音频水印设计中增强安全性的必要性。



## **38. DASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples**

DASH：一个用于合成有效且隐蔽的对抗示例的元攻击框架 cs.CV

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2508.13309v2) [paper-pdf](https://arxiv.org/pdf/2508.13309v2)

**Authors**: Abdullah Al Nomaan Nafi, Habibur Rahaman, Zafaryab Haider, Tanzim Mahfuz, Fnu Suya, Swarup Bhunia, Prabuddha Chakraborty

**Abstract**: Numerous techniques have been proposed for generating adversarial examples in white-box settings under strict Lp-norm constraints. However, such norm-bounded examples often fail to align well with human perception, and only recently have a few methods begun specifically exploring perceptually aligned adversarial examples. Moreover, it remains unclear whether insights from Lp-constrained attacks can be effectively leveraged to improve perceptual efficacy. In this paper, we introduce DAASH, a fully differentiable meta-attack framework that generates effective and perceptually aligned adversarial examples by strategically composing existing Lp-based attack methods. DAASH operates in a multi-stage fashion: at each stage, it aggregates candidate adversarial examples from multiple base attacks using learned, adaptive weights and propagates the result to the next stage. A novel meta-loss function guides this process by jointly minimizing misclassification loss and perceptual distortion, enabling the framework to dynamically modulate the contribution of each base attack throughout the stages. We evaluate DAASH on adversarially trained models across CIFAR-10, CIFAR-100, and ImageNet. Despite relying solely on Lp-constrained based methods, DAASH significantly outperforms state-of-the-art perceptual attacks such as AdvAD -- achieving higher attack success rates (e.g., 20.63\% improvement) and superior visual quality, as measured by SSIM, LPIPS, and FID (improvements $\approx$ of 11, 0.015, and 5.7, respectively). Furthermore, DAASH generalizes well to unseen defenses, making it a practical and strong baseline for evaluating robustness without requiring handcrafted adaptive attacks for each new defense.

摘要: 已经提出了许多技术来在严格的Lp范数约束下在白盒设置中生成对抗性示例。然而，这样的范数有界的例子往往不能很好地与人类的感知保持一致，直到最近才有一些方法开始专门探索感知对齐的对抗性例子。此外，目前还不清楚是否可以有效地利用Lp约束攻击的见解来提高感知效能。在本文中，我们介绍了DAASH，这是一个完全可区分的元攻击框架，它通过战略性地组合现有的基于LP的攻击方法来生成有效且感知一致的对抗性示例。DAASH以多阶段的方式运行：在每个阶段，它使用学习的自适应权重聚合来自多个基础攻击的候选对抗示例，并将结果传播到下一阶段。一种新颖的元损失函数通过联合最大限度地减少误分类损失和感知失真来指导这一过程，使框架能够动态调节整个阶段每个碱基攻击的贡献。我们在CIFAR-10、CIFAR-100和ImageNet中的对抗训练模型上评估DAASH。尽管仅依赖于基于LP约束的方法，DAASH的性能显着优于AdvAD等最先进的感知攻击--实现更高的攻击成功率（例如，改善20.63%）和优异的视觉质量，通过SSIM、LPIPS和DID衡量（改善$\约为11、0.015和5.7）。此外，DAASH很好地推广到了看不见的防御，使其成为评估稳健性的实用且强大的基线，而无需对每个新防御进行手工设计的自适应攻击。



## **39. How Not to Detect Prompt Injections with an LLM**

如何不使用LLM检测提示注射 cs.CR

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2507.05630v3) [paper-pdf](https://arxiv.org/pdf/2507.05630v3)

**Authors**: Sarthak Choudhary, Divyam Anshumaan, Nils Palumbo, Somesh Jha

**Abstract**: LLM-integrated applications and agents are vulnerable to prompt injection attacks, where adversaries embed malicious instructions within seemingly benign input data to manipulate the LLM's intended behavior. Recent defenses based on known-answer detection (KAD) scheme have reported near-perfect performance by observing an LLM's output to classify input data as clean or contaminated. KAD attempts to repurpose the very susceptibility to prompt injection as a defensive mechanism. We formally characterize the KAD scheme and uncover a structural vulnerability that invalidates its core security premise. To exploit this fundamental vulnerability, we methodically design an adaptive attack, DataFlip. It consistently evades KAD defenses, achieving detection rates as low as $0\%$ while reliably inducing malicious behavior with a success rate of $91\%$, all without requiring white-box access to the LLM or any optimization procedures.

摘要: LLM集成的应用程序和代理很容易受到提示注入攻击，对手将恶意指令嵌入看似良性的输入数据中，以操纵LLM的预期行为。最近基于已知答案检测（KAD）方案的防御报告称，通过观察LLM的输出将输入数据分类为干净或受污染，具有近乎完美的性能。KAD试图将即时注射的易感性重新利用为一种防御机制。我们正式描述了KAD方案的特征，并发现了使其核心安全前提无效的结构性漏洞。为了利用这个基本漏洞，我们有条不紊地设计了一种自适应攻击DataFlip。它始终规避KAD防御，实现低至0美元的检测率，同时以91美元的成功率可靠地诱导恶意行为，所有这些都不需要白盒访问LLM或任何优化程序。



## **40. Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model**

分析二进制等PDF：通过中间表示和语言模型进行对抗稳健的PDF恶意软件分析 cs.CR

Accepted by ACM CCS 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2506.17162v2) [paper-pdf](https://arxiv.org/pdf/2506.17162v2)

**Authors**: Side Liu, Jiang Ming, Guodong Zhou, Xinyi Liu, Jianming Fu, Guojun Peng

**Abstract**: Malicious PDF files have emerged as a persistent threat and become a popular attack vector in web-based attacks. While machine learning-based PDF malware classifiers have shown promise, these classifiers are often susceptible to adversarial attacks, undermining their reliability. To address this issue, recent studies have aimed to enhance the robustness of PDF classifiers. Despite these efforts, the feature engineering underlying these studies remains outdated. Consequently, even with the application of cutting-edge machine learning techniques, these approaches fail to fundamentally resolve the issue of feature instability.   To tackle this, we propose a novel approach for PDF feature extraction and PDF malware detection. We introduce the PDFObj IR (PDF Object Intermediate Representation), an assembly-like language framework for PDF objects, from which we extract semantic features using a pretrained language model. Additionally, we construct an Object Reference Graph to capture structural features, drawing inspiration from program analysis. This dual approach enables us to analyze and detect PDF malware based on both semantic and structural features. Experimental results demonstrate that our proposed classifier achieves strong adversarial robustness while maintaining an exceptionally low false positive rate of only 0.07% on baseline dataset compared to state-of-the-art PDF malware classifiers.

摘要: 恶意PDF文件已经成为一种持续的威胁，并成为基于Web的攻击中流行的攻击向量。虽然基于机器学习的PDF恶意软件分类器已经显示出前景，但这些分类器通常容易受到对抗性攻击，从而破坏了它们的可靠性。为了解决这个问题，最近的研究旨在提高PDF分类器的鲁棒性。尽管有这些努力，这些研究背后的特征工程仍然过时。因此，即使应用尖端的机器学习技术，这些方法也无法从根本上解决特征不稳定性问题。   为了解决这个问题，我们提出了一种新颖的PDF特征提取和PDF恶意软件检测方法。我们引入了PDFObj IR（PDF对象中间表示），这是一种用于PDF对象的类似汇编的语言框架，我们使用预先训练的语言模型从中提取语义特征。此外，我们还构建了一个对象引用图来捕获结构特征，从程序分析中汲取灵感。这种双重方法使我们能够根据语义和结构特征分析和检测PDF恶意软件。实验结果表明，我们提出的分类器实现了强大的对抗鲁棒性，同时保持了异常低的误报率只有0.07%的基线数据集相比，最先进的PDF恶意软件分类。



## **41. Exploring Adversarial Watermarking in Transformer-Based Models: Transferability and Robustness Against Defense Mechanism for Medical Images**

探索基于转换器的模型中的对抗性水印：医学图像防御机制的可移植性和鲁棒性 cs.CV

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2506.06389v2) [paper-pdf](https://arxiv.org/pdf/2506.06389v2)

**Authors**: Rifat Sadik, Tanvir Rahman, Arpan Bhattacharjee, Bikash Chandra Halder, Ismail Hossain, Rifat Sarker Aoyon, Md. Golam Rabiul Alam, Jia Uddin

**Abstract**: Deep learning models have shown remarkable success in dermatological image analysis, offering potential for automated skin disease diagnosis. Previously, convolutional neural network(CNN) based architectures have achieved immense popularity and success in computer vision (CV) based task like skin image recognition, generation and video analysis. But with the emergence of transformer based models, CV tasks are now are nowadays carrying out using these models. Vision Transformers (ViTs) is such a transformer-based models that have shown success in computer vision. It uses self-attention mechanisms to achieve state-of-the-art performance across various tasks. However, their reliance on global attention mechanisms makes them susceptible to adversarial perturbations. This paper aims to investigate the susceptibility of ViTs for medical images to adversarial watermarking-a method that adds so-called imperceptible perturbations in order to fool models. By generating adversarial watermarks through Projected Gradient Descent (PGD), we examine the transferability of such attacks to CNNs and analyze the performance defense mechanism -- adversarial training. Results indicate that while performance is not compromised for clean images, ViTs certainly become much more vulnerable to adversarial attacks: an accuracy drop of as low as 27.6%. Nevertheless, adversarial training raises it up to 90.0%.

摘要: 深度学习模型在皮肤病学图像分析方面取得了显着的成功，为自动化皮肤病诊断提供了潜力。此前，基于卷积神经网络（CNN）的架构在基于计算机视觉（CV）的任务（例如皮肤图像识别、生成和视频分析）中获得了巨大的普及和成功。但随着基于Transformer的模型的出现，CV任务现在正在使用这些模型执行。Vision Transformers（ViTS）是一种基于Transformers的模型，在计算机视觉领域取得了成功。它使用自我关注机制来在各种任务中实现最先进的性能。然而，它们对全球注意力机制的依赖使它们容易受到对抗性干扰。本文旨在研究医学图像的ViT对对抗性水印的敏感性--一种添加所谓的不可感知扰动以欺骗模型的方法。通过通过投影梯度下降（PVD）生成对抗性水印，我们检查了此类攻击到CNN的可转移性，并分析了性能防御机制--对抗性训练。结果表明，虽然干净图像的性能不会受到影响，但ViT确实变得更容易受到对抗攻击：准确性下降低至27.6%。尽管如此，对抗性训练将其提高至90.0%。



## **42. Unlearning Inversion Attacks for Graph Neural Networks**

消除图神经网络的反转攻击 cs.LG

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2506.00808v2) [paper-pdf](https://arxiv.org/pdf/2506.00808v2)

**Authors**: Jiahao Zhang, Yilong Wang, Zhiwei Zhang, Xiaorui Liu, Suhang Wang

**Abstract**: Graph unlearning methods aim to efficiently remove the impact of sensitive data from trained GNNs without full retraining, assuming that deleted information cannot be recovered. In this work, we challenge this assumption by introducing the graph unlearning inversion attack: given only black-box access to an unlearned GNN and partial graph knowledge, can an adversary reconstruct the removed edges? We identify two key challenges: varying probability-similarity thresholds for unlearned versus retained edges, and the difficulty of locating unlearned edge endpoints, and address them with TrendAttack. First, we derive and exploit the confidence pitfall, a theoretical and empirical pattern showing that nodes adjacent to unlearned edges exhibit a large drop in model confidence. Second, we design an adaptive prediction mechanism that applies different similarity thresholds to unlearned and other membership edges. Our framework flexibly integrates existing membership inference techniques and extends them with trend features. Experiments on four real-world datasets demonstrate that TrendAttack significantly outperforms state-of-the-art GNN membership inference baselines, exposing a critical privacy vulnerability in current graph unlearning methods.

摘要: 图去学习方法的目的是在不进行全面重新训练的情况下，有效地消除经过训练的GNN中敏感数据的影响，前提是无法恢复已删除的信息。在这项工作中，我们通过引入图未学习倒置攻击来挑战这一假设：仅在黑匣子访问未学习的GNN和部分图知识的情况下，对手能否重建被删除的边？我们确定了两个关键挑战：未学习边缘与保留边缘的概率相似性阈值不同，以及定位未学习边缘端点的困难，并使用TrendAttack解决这些问题。首先，我们推导并利用置信陷阱，这是一种理论和经验模式，表明邻近未学习边的节点表现出模型置信度大幅下降。其次，我们设计了一个自适应的预测机制，适用于不同的相似性阈值的未学习和其他成员的边缘。我们的框架灵活地集成了现有的成员推理技术，并扩展它们的趋势功能。在四个真实世界数据集上的实验表明，TrendAttack的性能明显优于最先进的GNN成员推理基线，暴露了当前图形学习方法中的关键隐私漏洞。



## **43. Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains**

时间序列基础模型是否已准备好部署？跨领域对抗稳健性的系统研究 cs.LG

Preprint

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2505.19397v2) [paper-pdf](https://arxiv.org/pdf/2505.19397v2)

**Authors**: Jiawen Zhang, Zhenwei Zhang, Shun Zheng, Xumeng Wen, Jia Li, Jiang Bian

**Abstract**: Time-Series Foundation Models (TSFMs) are rapidly transitioning from research prototypes to core components of critical decision-making systems, driven by their impressive zero-shot forecasting capabilities. However, as their deployment surges, a critical blind spot remains: their fragility under adversarial attacks. This lack of scrutiny poses severe risks, particularly as TSFMs enter high-stakes environments vulnerable to manipulation. We present a systematic, diagnostic study arguing that for TSFMs, robustness is not merely a secondary metric but a prerequisite for trustworthy deployment comparable to accuracy. Our evaluation framework, explicitly tailored to the unique constraints of time series, incorporates normalized, sparsity-aware perturbation budgets and unified scale-invariant metrics across white-box and black-box settings. Across six representative TSFMs, we demonstrate that current architectures are alarmingly brittle: even small perturbations can reliably steer forecasts toward specific failure modes, such as trend flips and malicious drifts. We uncover TSFM-specific vulnerability patterns, including horizon-proximal brittleness, increased susceptibility with longer context windows, and weak cross-model transfer that points to model-specific failure modes rather than generic distortions. Finally, we show that simple adversarial fine-tuning offers a cost-effective path to substantial robustness gains, even with out-of-domain data. This work bridges the gap between TSFM capabilities and safety constraints, offering essential guidance for hardening the next generation of forecasting systems.

摘要: 时间序列基础模型（TSFM）在其令人印象深刻的零冲击预测能力的推动下，正在从研究原型迅速过渡到关键决策系统的核心组件。然而，随着它们部署的激增，一个关键的盲点仍然存在：它们在敌对攻击下的脆弱性。这种缺乏审查带来了严重的风险，特别是当TSFM进入容易受到操纵的高风险环境时。我们提出了一项系统性的诊断研究，认为对于TSFM来说，稳健性不仅是次要指标，而且是与准确性相当的值得信赖部署的先决条件。我们的评估框架明确针对时间序列的独特约束进行定制，融合了规范化、稀疏性感知的扰动预算和跨白盒和黑箱设置的统一尺度不变指标。在六个有代表性的TSFM中，我们证明了当前的架构非常脆弱：即使是很小的扰动也可以可靠地将预测引导到特定的故障模式，例如趋势翻转和恶意漂移。我们发现了特定于TFM的漏洞模式，包括水平近端脆性、更长上下文窗口的易感性增加以及指向特定于模型的故障模式而不是通用扭曲的弱跨模型转移。最后，我们表明，即使使用域外数据，简单的对抗性微调也可以提供一条经济有效的途径来获得可观的鲁棒性。这项工作弥合了TSFM能力和安全限制之间的差距，为强化下一代预测系统提供了重要指导。



## **44. Evaluating the robustness of adversarial defenses in malware detection systems**

评估恶意软件检测系统中对抗防御的稳健性 cs.CR

Published in Computers & Electrical Engineering (Elsevier), Volume 130, February 2026, Article 110845

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2505.09342v2) [paper-pdf](https://arxiv.org/pdf/2505.09342v2)

**Authors**: Mostafa Jafari, Alireza Shameli-Sendi

**Abstract**: Machine learning is a key tool for Android malware detection, effectively identifying malicious patterns in apps. However, ML-based detectors are vulnerable to evasion attacks, where small, crafted changes bypass detection. Despite progress in adversarial defenses, the lack of comprehensive evaluation frameworks in binary-constrained domains limits understanding of their robustness. We introduce two key contributions. First, Prioritized Binary Rounding, a technique to convert continuous perturbations into binary feature spaces while preserving high attack success and low perturbation size. Second, the sigma-binary attack, a novel adversarial method for binary domains, designed to achieve attack goals with minimal feature changes. Experiments on the Malscan dataset show that sigma-binary outperforms existing attacks and exposes key vulnerabilities in state-of-the-art defenses. Defenses equipped with adversary detectors, such as KDE, DLA, DNN+, and ICNN, exhibit significant brittleness, with attack success rates exceeding 90% using fewer than 10 feature modifications and reaching 100% with just 20. Adversarially trained defenses, including AT-rFGSM-k, AT-MaxMA, improves robustness under small budgets but remains vulnerable to unrestricted perturbations, with attack success rates of 99.45% and 96.62%, respectively. Although PAD-SMA demonstrates strong robustness against state-of-the-art gradient-based adversarial attacks by maintaining an attack success rate below 16.55%, the sigma-binary attack significantly outperforms these methods, achieving a 94.56% success rate under unrestricted perturbations. These findings highlight the critical need for precise method like sigma-binary to expose hidden vulnerabilities in existing defenses and support the development of more resilient malware detection systems.

摘要: 机器学习是Android恶意软件检测的关键工具，可以有效识别应用程序中的恶意模式。然而，基于ML的检测器很容易受到规避攻击，因为小的精心设计的更改会绕过检测。尽管对抗性防御取得了进展，但二进制约束领域缺乏全面的评估框架限制了对其稳健性的理解。我们介绍两个关键贡献。首先，优先二进制舍入，这是一种将连续扰动转换为二进制特征空间的技术，同时保持高攻击成功率和低扰动大小。其次，西格玛二进制攻击，这是一种针对二进制域的新型对抗方法，旨在以最小的特征变化实现攻击目标。Malcan数据集的实验表明，西格玛二进制优于现有攻击，并暴露了最先进防御中的关键漏洞。配备对手检测器的防御系统，例如TEK、DLA、DNN+和ICNN，表现出显着的脆性，使用少于10个功能修改，攻击成功率超过90%，仅使用20个功能修改即可达到100%。经过对抗训练的防御，包括AT-rFGSM-k、AT-MaxMA，可以在小预算下提高稳健性，但仍然容易受到不受限制的干扰，攻击成功率分别为99.45%和96.62%。尽管PAD-SM通过将攻击成功率保持在16.55%以下，表现出对最先进的基于梯度的对抗攻击的强大鲁棒性，但西格玛二进制攻击的表现显着优于这些方法，在不受限制的扰动下实现了94.56%的成功率。这些发现凸显了迫切需要西格玛二进制等精确方法来暴露现有防御中隐藏的漏洞并支持开发更具弹性的恶意软件检测系统。



## **45. Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots**

探索自主移动机器人基于搜索的路径规划中的对抗障碍攻击 cs.RO

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2504.06154v2) [paper-pdf](https://arxiv.org/pdf/2504.06154v2)

**Authors**: Adrian Szvoren, Jianwei Liu, Dimitrios Kanoulas, Nilufer Tuptuk

**Abstract**: Path planning algorithms, such as the search-based A*, are a critical component of autonomous mobile robotics, enabling robots to navigate from a starting point to a destination efficiently and safely. We investigated the resilience of the A* algorithm in the face of potential adversarial interventions known as obstacle attacks. The adversary's goal is to delay the robot's timely arrival at its destination by introducing obstacles along its original path.   We developed malicious software to execute the attacks and conducted experiments to assess their impact, both in simulation using TurtleBot in Gazebo and in real-world deployment with the Unitree Go1 robot. In simulation, the attacks resulted in an average delay of 36\%, with the most significant delays occurring in scenarios where the robot was forced to take substantially longer alternative paths. In real-world experiments, the delays were even more pronounced, with all attacks successfully rerouting the robot and causing measurable disruptions. These results highlight that the algorithm's robustness is not solely an attribute of its design but is significantly influenced by the operational environment. For example, in constrained environments like tunnels, the delays were maximized due to the limited availability of alternative routes.

摘要: 路径规划算法（例如基于搜索的A*）是自主移动机器人技术的关键组成部分，使机器人能够高效、安全地从起点导航到目的地。我们研究了A* 算法在面临潜在的对抗性干预（即障碍攻击）时的弹性。对手的目标是通过在机器人的原始路径上设置障碍物来推迟机器人及时到达目的地。   我们开发了恶意软件来执行攻击，并进行了实验来评估其影响，无论是在Gazebo中使用TurtleBot进行模拟还是在现实世界中使用Unitree Go 1机器人进行部署。在模拟中，攻击导致平均延迟为36%，其中最显着的延迟发生在机器人被迫采取更长的替代路径的情况下。在现实世界的实验中，延迟甚至更加明显，所有攻击都成功地改变了机器人的路线并造成了可测量的中断。这些结果凸显了该算法的鲁棒性不仅仅是其设计的属性，而且还受到操作环境的显着影响。例如，在隧道等受限环境中，由于替代路线的可用性有限，延误被最大化。



## **46. Causal Interpretability for Adversarial Robustness: A Hybrid Generative Classification Approach**

对抗稳健性的因果解释性：混合生成分类方法 cs.CV

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2412.20025v2) [paper-pdf](https://arxiv.org/pdf/2412.20025v2)

**Authors**: Chunheng Zhao, Pierluigi Pisu, Gurcan Comert, Negash Begashaw, Varghese Vaidyan, Nina Christine Hubig

**Abstract**: Deep learning-based discriminative classifiers, despite their remarkable success, remain vulnerable to adversarial examples that can mislead model predictions. While adversarial training can enhance robustness, it fails to address the intrinsic vulnerability stemming from the opaque nature of these black-box models. We present a deep ensemble model that combines discriminative features with generative models to achieve both high accuracy and adversarial robustness. Our approach integrates a bottom-level pre-trained discriminative network for feature extraction with a top-level generative classification network that models adversarial input distributions through a deep latent variable model. Using variational Bayes, our model achieves superior robustness against white-box adversarial attacks without adversarial training. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate our model's superior adversarial robustness. Through evaluations using counterfactual metrics and feature interaction-based metrics, we establish correlations between model interpretability and adversarial robustness. Additionally, preliminary results on Tiny-ImageNet validate our approach's scalability to more complex datasets, offering a practical solution for developing robust image classification models.

摘要: 基于深度学习的区分分类器尽管取得了显着的成功，但仍然容易受到可能误导模型预测的对抗示例的影响。虽然对抗性训练可以增强稳健性，但它未能解决这些黑匣子模型不透明性质所带来的内在脆弱性。我们提出了一种深度集成模型，将区分特征与生成模型相结合，以实现高准确性和对抗鲁棒性。我们的方法将用于特征提取的底层预训练辨别网络与顶层生成分类网络集成，该网络通过深度潜在变量模型对对抗性输入分布进行建模。使用变分Bayes，我们的模型无需对抗训练即可获得针对白盒对抗攻击的卓越鲁棒性。对CIFAR-10和CIFAR-100的大量实验证明了我们的模型具有卓越的对抗鲁棒性。通过使用反事实指标和基于特征交互的指标进行评估，我们建立了模型可解释性和对抗稳健性之间的相关性。此外，Tiny-ImageNet的初步结果验证了我们的方法对更复杂数据集的可扩展性，为开发稳健的图像分类模型提供了实用的解决方案。



## **47. AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models**

AEIOU：针对文本到图像模型中NSFW格式的统一防御框架 cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2412.18123v3) [paper-pdf](https://arxiv.org/pdf/2412.18123v3)

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Tong Zhang, Rui Zeng, Xing Yang, Shouling Ji

**Abstract**: As text-to-image (T2I) models advance and gain widespread adoption, their associated safety concerns are becoming increasingly critical. Malicious users exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, underscoring the need for effective safeguards to ensure the integrity and compliance of model outputs. However, existing detection methods often exhibit low accuracy and inefficiency. In this paper, we propose AEIOU, a defense framework that is adaptable, efficient, interpretable, optimizable, and unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95\% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.

摘要: 随着文本到图像（T2 I）模型的发展和广泛采用，其相关的安全问题变得越来越严重。恶意用户利用这些模型使用有害或对抗性提示生成不安全工作（NSFW）图像，这凸显了有效的保障措施以确保模型输出的完整性和合规性的必要性。然而，现有的检测方法往往表现出准确性低且效率低下。在本文中，我们提出AEIOU，这是一种防御框架，具有适应性、高效性、可解释性、可优化性，并且针对T2 I模型中的NSFW提示进行统一。AEIOU从模型文本编码器的隐藏状态中提取NSFW特征，利用这些特征的可分离性来检测NSFW提示。检测过程是高效的，需要最少的推理时间。AEIOU还提供结果的实时解释，并通过数据增强技术支持优化。该框架是通用的，可适应各种T2 I架构。我们广泛的实验表明，AEIOU的性能显着优于商业和开源审核工具，在所有数据集中实现了超过95%的准确性，并将效率提高了至少十倍。它有效地对抗自适应攻击，并在少数镜头和多标签场景中表现出色。



## **48. GLL: A Differentiable Graph Learning Layer for Neural Networks**

GLL：神经网络的可区分图学习层 cs.LG

58 pages, 12 figures. Preprint. Submitted to the Journal of Machine Learning Research. v2: several new experiments, improved exposition

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2412.08016v2) [paper-pdf](https://arxiv.org/pdf/2412.08016v2)

**Authors**: Jason Brown, Bohan Chen, Harris Hardiman-Mostow, Jeff Calder, Andrea L. Bertozzi

**Abstract**: Standard deep learning architectures used for classification generate label predictions with a projection head and softmax activation function. Although successful, these methods fail to leverage the relational information between samples for generating label predictions. In recent works, graph-based learning techniques, namely Laplace learning, have been heuristically combined with neural networks for both supervised and semi-supervised learning (SSL) tasks. However, prior works approximate the gradient of the loss function with respect to the graph learning algorithm or decouple the processes; end-to-end integration with neural networks is not achieved. In this work, we derive backpropagation equations, via the adjoint method, for inclusion of a general family of graph learning layers into a neural network. The resulting method, distinct from graph neural networks, allows us to precisely integrate similarity graph construction and graph Laplacian-based label propagation into a neural network layer, replacing a projection head and softmax activation function for general classification task. Our experimental results demonstrate smooth label transitions across data, improved generalization and robustness to adversarial attacks, and improved training dynamics compared to a standard softmax-based approach.

摘要: 用于分类的标准深度学习架构通过投影头和softmax激活功能生成标签预测。尽管成功，但这些方法未能利用样本之间的关系信息来生成标签预测。在最近的工作中，基于图的学习技术（即拉普拉斯学习）已与神经网络进行了逻辑性地结合，用于监督和半监督学习（SSL）任务。然而，先前的工作接近损失函数相对于图学习算法的梯度或将过程脱钩;未实现与神经网络的端到端集成。在这项工作中，我们通过伴随方法推导出反向传播方程，以便将一般的图学习层家族纳入神经网络。由此产生的方法与图神经网络不同，使我们能够将相似性图构建和基于图拉普拉斯的标签传播精确集成到神经网络层中，取代一般分类任务的投影头和softmax激活函数。我们的实验结果表明，与标准的基于softmax的方法相比，数据之间的标签转换平滑、对对抗攻击的概括性和鲁棒性得到提高，以及训练动态得到改进。



## **49. Edge-Only Universal Adversarial Attacks in Distributed Learning**

分布式学习中的仅边通用对抗攻击 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2411.10500v2) [paper-pdf](https://arxiv.org/pdf/2411.10500v2)

**Authors**: Giulio Rossolini, Tommaso Baldi, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: Distributed learning frameworks, which partition neural network models across multiple computing nodes, enhance efficiency in collaborative edge-cloud systems, but may also introduce new vulnerabilities to evasion attacks, often in the form of adversarial perturbations. In this work, we present a new threat model that explores the feasibility of generating universal adversarial perturbations (UAPs) when the attacker has access only to the edge portion of the model, consisting of its initial network layers. Unlike traditional attacks that require full model knowledge, our approach shows that adversaries can induce effective mispredictions in the unknown cloud component by manipulating key feature representations at the edge. Following the proposed threat model, we introduce both edge-only untargeted and targeted formulations of UAPs designed to control intermediate features before the split point. Our results on ImageNet demonstrate strong attack transferability to the unknown cloud part, and we compare the proposed method with classical white-box and black-box techniques, highlighting its effectiveness. Additionally, we analyze the capability of an attacker to achieve targeted adversarial effects with edge-only knowledge, revealing intriguing behaviors across multiple networks. By introducing the first adversarial attacks with edge-only knowledge in split inference, this work underscores the importance of addressing partial model access in adversarial robustness, encouraging further research in this area.

摘要: 分布式学习框架将神经网络模型划分为多个计算节点，可以提高协作边缘云系统的效率，但也可能为规避攻击引入新的漏洞，通常以对抗性扰动的形式。在这项工作中，我们提出了一种新的威胁模型，该模型探讨了当攻击者只能访问模型的边缘部分（由其初始网络层组成）时生成普遍对抗扰动（UPC）的可行性。与需要完整模型知识的传统攻击不同，我们的方法表明，对手可以通过操纵边缘的关键特征表示来在未知云组件中引发有效的误预测。遵循拟议的威胁模型，我们引入了仅边缘非定向和定向的UAP公式，旨在控制分裂点之前的中间特征。我们在ImageNet上的结果证明了攻击对未知云部分的强大可转移性，并且我们将提出的方法与经典的白盒和黑盒技术进行了比较，强调了其有效性。此外，我们还分析了攻击者利用仅边缘知识实现有针对性的对抗效果的能力，揭示了多个网络中有趣的行为。通过在分裂推理中引入第一个具有仅边知识的对抗攻击，这项工作强调了解决部分模型访问在对抗稳健性中的重要性，鼓励了该领域的进一步研究。



## **50. Attacking All Tasks at Once Using Adversarial Examples in Multi-Task Learning**

在多任务学习中使用对抗示例同时攻击所有任务 cs.LG

Accpeted by Neurocomputing at 10 September 2025

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2305.12066v4) [paper-pdf](https://arxiv.org/pdf/2305.12066v4)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Visual content understanding frequently relies on multi-task models to extract robust representations of a single visual input for multiple downstream tasks. However, in comparison to extensively studied single-task models, the adversarial robustness of multi-task models has received significantly less attention and many questions remain unclear: 1) How robust are multi-task models to single task adversarial attacks, 2) Can adversarial attacks be designed to simultaneously attack all tasks in a multi-task model, and 3) How does parameter sharing across tasks affect multi-task model robustness to adversarial attacks? This paper aims to answer these questions through careful analysis and rigorous experimentation. First, we analyze the inherent drawbacks of two commonly-used adaptations of single-task white-box attacks in attacking multi-task models. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking all tasks in a multi-task model as an optimization problem that can be efficiently solved through integer linear programming. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to baselines in attacking both clean and adversarially trained multi-task models. Our results also reveal a fundamental trade-off between improving task accuracy via parameter sharing across tasks and undermining model robustness due to increased attack transferability from parameter sharing.

摘要: 视觉内容理解通常依赖于多任务模型来为多个下游任务提取单个视觉输入的稳健表示。然而，与广泛研究的单任务模型相比，多任务模型的对抗稳健性受到的关注明显较少，许多问题仍然不清楚：1）多任务模型对单任务对抗攻击的鲁棒性有多强，2）对抗攻击能否被设计为同时攻击多任务模型中的所有任务，3）任务之间的参数共享如何影响多任务模型对对抗攻击的稳健性？本文旨在通过仔细的分析和严格的实验来回答这些问题。首先，我们分析了两种常用的单任务白盒攻击改编方法在攻击多任务模型时的固有缺陷。然后，我们提出了一种新颖的攻击框架：动态梯度平衡攻击（DGBA）。我们的框架将攻击多任务模型中所有任务的问题提出为优化问题，可以通过整线性规划有效地解决。对两个流行的MTL基准NYUv 2和Tiny-Taxonomy的广泛评估表明，与基线相比，DGBA在攻击干净和对抗训练的多任务模型方面的有效性。我们的结果还揭示了通过任务之间的参数共享提高任务准确性与由于参数共享增加攻击可转移性而破坏模型稳健性之间的根本权衡。



