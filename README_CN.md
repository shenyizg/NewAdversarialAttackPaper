# Latest Adversarial Attack Papers
**update at 2025-12-16 18:37:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. REVERB-FL: Server-Side Adversarial and Reserve-Enhanced Federated Learning for Robust Audio Classification**

REVERB-FL：用于稳健音频分类的服务器端对抗和保留增强联邦学习 eess.AS

13 pages, 4 figures

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13647v1) [paper-pdf](https://arxiv.org/pdf/2512.13647v1)

**Authors**: Sathwika Peechara, Rajeev Sahay

**Abstract**: Federated learning (FL) enables a privacy-preserving training paradigm for audio classification but is highly sensitive to client heterogeneity and poisoning attacks, where adversarially compromised clients can bias the global model and hinder the performance of audio classifiers. To mitigate the effects of model poisoning for audio signal classification, we present REVERB-FL, a lightweight, server-side defense that couples a small reserve set (approximately 5%) with pre- and post-aggregation retraining and adversarial training. After each local training round, the server refines the global model on the reserve set with either clean or additional adversarially perturbed data, thereby counteracting non-IID drift and mitigating potential model poisoning without adding substantial client-side cost or altering the aggregation process. We theoretically demonstrate the feasibility of our framework, showing faster convergence and a reduced steady-state error relative to baseline federated averaging. We validate our framework on two open-source audio classification datasets with varying IID and Dirichlet non-IID partitions and demonstrate that REVERB-FL mitigates global model poisoning under multiple designs of local data poisoning.

摘要: 联合学习（FL）为音频分类提供了一种保护隐私的训练范式，但对客户端的多样性和中毒攻击高度敏感，其中敌对妥协的客户端可能会对全局模型产生偏差并阻碍音频分类器的性能。为了减轻模型中毒对音频信号分类的影响，我们提出了REVERB-FL，这是一种轻量级的服务器端防御，它将小的储备集（约5%）与聚集前和聚集后的再训练和对抗训练结合起来。每次本地训练轮结束后，服务器都会用干净的或额外的敌对扰动数据来细化储备集中的全局模型，从而抵消非IID漂移并减轻潜在的模型中毒，而不会增加大量客户端成本或改变聚合过程。我们从理论上证明了我们框架的可行性，相对于基线联邦平均，收敛速度更快，稳态误差更小。我们在具有不同IID和Dirichlet非IID分区的两个开源音频分类数据集上验证了我们的框架，并证明REVERB-FL可以缓解多种本地数据中毒设计下的全局模型中毒。



## **2. Async Control: Stress-testing Asynchronous Control Measures for LLM Agents**

同步控制：LLM代理的压力测试同步控制措施 cs.LG

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13526v1) [paper-pdf](https://arxiv.org/pdf/2512.13526v1)

**Authors**: Asa Cooper Stickland, Jan Michelfeit, Arathi Mani, Charlie Griffin, Ollie Matthews, Tomek Korbak, Rogan Inglis, Oliver Makins, Alan Cooney

**Abstract**: LLM-based software engineering agents are increasingly used in real-world development tasks, often with access to sensitive data or security-critical codebases. Such agents could intentionally sabotage these codebases if they were misaligned. We investigate asynchronous monitoring, in which a monitoring system reviews agent actions after the fact. Unlike synchronous monitoring, this approach does not impose runtime latency, while still attempting to disrupt attacks before irreversible harm occurs. We treat monitor development as an adversarial game between a blue team (who design monitors) and a red team (who create sabotaging agents). We attempt to set the game rules such that they upper bound the sabotage potential of an agent based on Claude 4.1 Opus. To ground this game in a realistic, high-stakes deployment scenario, we develop a suite of 5 diverse software engineering environments that simulate tasks that an agent might perform within an AI developer's internal infrastructure. Over the course of the game, we develop an ensemble monitor that achieves a 6% false negative rate at 1% false positive rate on a held out test environment. Then, we estimate risk of sabotage at deployment time by extrapolating from our monitor's false negative rate. We describe one simple model for this extrapolation, present a sensitivity analysis, and describe situations in which the model would be invalid. Code is available at: https://github.com/UKGovernmentBEIS/async-control.

摘要: 基于LLM的软件工程代理越来越多地用于现实世界的开发任务，通常可以访问敏感数据或安全关键代码库。如果这些代码库未对齐，此类代理可能会故意破坏它们。我们研究了同步监控，其中监控系统在事后审查代理动作。与同步监控不同，这种方法不会施加运行时延迟，同时仍试图在不可逆转的伤害发生之前中断攻击。我们将显示器开发视为蓝色团队（设计显示器）和红色团队（创建破坏性代理）之间的对抗游戏。我们试图设定游戏规则，使其上限基于Claude 4.1 Opus的代理的破坏潜力。为了将这款游戏置于现实、高风险的部署场景中，我们开发了一套由5个不同的软件工程环境，这些环境模拟代理可能在人工智能开发人员的内部基础设施中执行的任务。在游戏过程中，我们开发了一个整体监视器，可以在固定的测试环境中实现6%的假阴性率和1%的假阳性率。然后，我们通过从监视器的假阴性率推断来估计部署时破坏的风险。我们描述了一个用于此外推的简单模型，提出了敏感性分析，并描述了模型无效的情况。代码可访问：https://github.com/UKGovernmentBEIS/async-control。



## **3. An $H_2$-norm approach to performance analysis of networked control systems under multiplicative routing transformations**

乘性路由变换下网络控制系统性能分析的$H_2$-模方法 eess.SY

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13504v1) [paper-pdf](https://arxiv.org/pdf/2512.13504v1)

**Authors**: Ruslan Seifullaev, André M. H. Teixeira

**Abstract**: This paper investigates the performance of networked control systems subject to multiplicative routing transformations that alter measurement pathways without directly injecting signals. Such transformations, arising from faults or adversarial actions, modify the feedback structure and can degrade performance while remaining stealthy. An $H_2$-norm framework is proposed to quantify the impact of these transformations by evaluating the ratio between the steady-state energies of performance and residual outputs. Equivalent linear matrix inequality (LMI) formulations are derived for computational assessment, and analytical upper bounds are established to estimate the worst-case degradation. The results provide structural insight into how routing manipulations influence closed-loop behavior and reveal conditions for stealthy multiplicative attacks.

摘要: 本文研究了经历乘性路由变换的网络控制系统的性能，该变换可以在不直接注入信号的情况下改变测量路径。这种由故障或对抗行为引起的转换会修改反馈结构，并可能会降低性能，同时保持隐形。提出了一个$H_2$-norm框架，通过评估性能的稳态能量与剩余产出之间的比率来量化这些转换的影响。推导出用于计算评估的等效线性矩阵不等式（LGA）公式，并建立分析上界以估计最坏情况的退化。结果提供了结构性的洞察路由操作如何影响闭环行为，并揭示了隐形乘法攻击的条件。



## **4. Behavior-Aware and Generalizable Defense Against Black-Box Adversarial Attacks for ML-Based IDS**

针对基于ML的IDS的黑匣子对抗攻击的行为感知和可推广防御 cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13501v1) [paper-pdf](https://arxiv.org/pdf/2512.13501v1)

**Authors**: Sabrine Ennaji, Elhadj Benkhelifa, Luigi Vincenzo Mancini

**Abstract**: Machine learning based intrusion detection systems are increasingly targeted by black box adversarial attacks, where attackers craft evasive inputs using indirect feedback such as binary outputs or behavioral signals like response time and resource usage. While several defenses have been proposed, including input transformation, adversarial training, and surrogate detection, they often fall short in practice. Most are tailored to specific attack types, require internal model access, or rely on static mechanisms that fail to generalize across evolving attack strategies. Furthermore, defenses such as input transformation can degrade intrusion detection system performance, making them unsuitable for real time deployment.   To address these limitations, we propose Adaptive Feature Poisoning, a lightweight and proactive defense mechanism designed specifically for realistic black box scenarios. Adaptive Feature Poisoning assumes that probing can occur silently and continuously, and introduces dynamic and context aware perturbations to selected traffic features, corrupting the attacker feedback loop without impacting detection capabilities. The method leverages traffic profiling, change point detection, and adaptive scaling to selectively perturb features that an attacker is likely exploiting, based on observed deviations.   We evaluate Adaptive Feature Poisoning against multiple realistic adversarial attack strategies, including silent probing, transferability based attacks, and decision boundary based attacks. The results demonstrate its ability to confuse attackers, degrade attack effectiveness, and preserve detection performance. By offering a generalizable, attack agnostic, and undetectable defense, Adaptive Feature Poisoning represents a significant step toward practical and robust adversarial resilience in machine learning based intrusion detection systems.

摘要: 基于机器学习的入侵检测系统越来越成为黑匣子对抗攻击的目标，攻击者使用间接反馈（例如二进制输出或响应时间和资源使用等行为信号）来制造规避输入。虽然已经提出了多种防御措施，包括输入转换、对抗训练和代理检测，但它们在实践中往往达不到要求。大多数都是针对特定攻击类型定制的，需要内部模型访问，或者依赖于无法在不断发展的攻击策略中进行概括的静态机制。此外，输入转换等防御措施可能会降低入侵检测系统的性能，使其不适合实时部署。   为了解决这些限制，我们提出了自适应特征中毒，这是一种专门为现实黑匣子场景设计的轻量级主动防御机制。自适应特征中毒假设探测可以无声且连续地发生，并向选定的流量特征引入动态和上下文感知的扰动，从而破坏攻击者反馈循环，而不影响检测能力。该方法利用流量分析、变点检测和自适应缩放来根据观察到的偏差选择性地扰乱攻击者可能利用的特征。   我们针对多种现实的对抗攻击策略来评估自适应特征中毒，包括无声探测、基于可转移性的攻击和基于决策边界的攻击。结果表明，它能够混淆攻击者，降低攻击效果，并保持检测性能。通过提供可推广的，攻击不可知的和不可检测的防御，自适应特征中毒代表了在基于机器学习的入侵检测系统中实现实用和强大的对抗弹性的重要一步。



## **5. On the Effectiveness of Membership Inference in Targeted Data Extraction from Large Language Models**

隶属推理在大型语言模型有针对性数据提取中的有效性 cs.LG

Accepted to IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13352v1) [paper-pdf](https://arxiv.org/pdf/2512.13352v1)

**Authors**: Ali Al Sahili, Ali Chehab, Razane Tajeddine

**Abstract**: Large Language Models (LLMs) are prone to memorizing training data, which poses serious privacy risks. Two of the most prominent concerns are training data extraction and Membership Inference Attacks (MIAs). Prior research has shown that these threats are interconnected: adversaries can extract training data from an LLM by querying the model to generate a large volume of text and subsequently applying MIAs to verify whether a particular data point was included in the training set. In this study, we integrate multiple MIA techniques into the data extraction pipeline to systematically benchmark their effectiveness. We then compare their performance in this integrated setting against results from conventional MIA benchmarks, allowing us to evaluate their practical utility in real-world extraction scenarios.

摘要: 大型语言模型（LLM）容易记住训练数据，这会带来严重的隐私风险。两个最突出的问题是训练数据提取和成员推断攻击（MIA）。之前的研究表明，这些威胁是相互关联的：对手可以通过查询模型以生成大量文本，然后应用MIA来验证特定数据点是否包含在训练集中，从而从LLM中提取训练数据。在这项研究中，我们将多种MIA技术集成到数据提取管道中，以系统地衡量其有效性。然后，我们将它们在此集成环境中的性能与传统MIA基准的结果进行比较，使我们能够评估它们在现实世界提取场景中的实际实用性。



## **6. Evaluating Adversarial Attacks on Federated Learning for Temperature Forecasting**

评估温度预测联邦学习的对抗攻击 cs.LG

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13207v1) [paper-pdf](https://arxiv.org/pdf/2512.13207v1)

**Authors**: Karina Chichifoi, Fabio Merizzi, Michele Colajanni

**Abstract**: Deep learning and federated learning (FL) are becoming powerful partners for next-generation weather forecasting. Deep learning enables high-resolution spatiotemporal forecasts that can surpass traditional numerical models, while FL allows institutions in different locations to collaboratively train models without sharing raw data, addressing efficiency and security concerns. While FL has shown promise across heterogeneous regions, its distributed nature introduces new vulnerabilities. In particular, data poisoning attacks, in which compromised clients inject manipulated training data, can degrade performance or introduce systematic biases. These threats are amplified by spatial dependencies in meteorological data, allowing localized perturbations to influence broader regions through global model aggregation. In this study, we investigate how adversarial clients distort federated surface temperature forecasts trained on the Copernicus European Regional ReAnalysis (CERRA) dataset. We simulate geographically distributed clients and evaluate patch-based and global biasing attacks on regional temperature forecasts. Our results show that even a small fraction of poisoned clients can mislead predictions across large, spatially connected areas. A global temperature bias attack from a single compromised client shifts predictions by up to -1.7 K, while coordinated patch attacks more than triple the mean squared error and produce persistent regional anomalies exceeding +3.5 K. Finally, we assess trimmed mean aggregation as a defense mechanism, showing that it successfully defends against global bias attacks (2-13\% degradation) but fails against patch attacks (281-603\% amplification), exposing limitations of outlier-based defenses for spatially correlated data.

摘要: 深度学习和联合学习（FL）正在成为下一代天气预报的强大合作伙伴。深度学习可以实现超越传统数值模型的高分辨率时空预测，而FL则允许不同地点的机构在无需共享原始数据的情况下协作训练模型，从而解决效率和安全问题。虽然FL在不同地区表现出了希望，但其分布式性质带来了新的漏洞。特别是，数据中毒攻击（即受影响的客户端注入操纵的训练数据）可能会降低性能或引入系统性偏差。气象数据的空间依赖性放大了这些威胁，使局部扰动能够通过全球模型聚合影响更广泛的区域。在这项研究中，我们调查了对抗性客户端如何扭曲在哥白尼欧洲区域再分析（CERRA）数据集上训练的联合表面温度预测。我们模拟地理上分布的客户端和评估补丁为基础的和全球偏见的攻击区域温度预报。我们的研究结果表明，即使是一小部分中毒的客户端也会误导大型空间连接区域的预测。来自单个受损客户端的全球温度偏差攻击将预测值改变高达-1.7 K，而协调补丁攻击则是均方误差的三倍多，并产生超过+3.5 K的持续区域异常。最后，我们评估了修剪平均聚集作为防御机制，表明它成功地防御全局偏差攻击（2- 13%降级），但未能防御补丁攻击（281- 603%放大），暴露了基于离群值的防御空间相关数据的局限性。



## **7. Less Is More: Sparse and Cooperative Perturbation for Point Cloud Attacks**

少即是多：点云攻击的稀疏和合作扰动 cs.CR

Accepted by AAAI'2026 (Oral)

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13119v1) [paper-pdf](https://arxiv.org/pdf/2512.13119v1)

**Authors**: Keke Tang, Tianyu Hao, Xiaofei Wang, Weilong Peng, Denghui Zhang, Peican Zhu, Zhihong Tian

**Abstract**: Most adversarial attacks on point clouds perturb a large number of points, causing widespread geometric changes and limiting applicability in real-world scenarios. While recent works explore sparse attacks by modifying only a few points, such approaches often struggle to maintain effectiveness due to the limited influence of individual perturbations. In this paper, we propose SCP, a sparse and cooperative perturbation framework that selects and leverages a compact subset of points whose joint perturbations produce amplified adversarial effects. Specifically, SCP identifies the subset where the misclassification loss is locally convex with respect to their joint perturbations, determined by checking the positivedefiniteness of the corresponding Hessian block. The selected subset is then optimized to generate high-impact adversarial examples with minimal modifications. Extensive experiments show that SCP achieves 100% attack success rates, surpassing state-of-the-art sparse attacks, and delivers superior imperceptibility to dense attacks with far fewer modifications.

摘要: 对点云的大多数对抗攻击都会扰乱大量点，导致广泛的几何变化并限制其在现实世界场景中的适用性。虽然最近的作品仅通过修改几个点来探索稀疏攻击，但由于个体扰动的影响有限，此类方法通常难以维持有效性。在本文中，我们提出了SCP，这是一种稀疏和合作的扰动框架，它选择和利用点的紧凑子集，这些点的联合扰动会产生放大的对抗效应。具体来说，SCP识别错误分类损失相对于其联合扰动为局部凸的子集，通过检查相应Hessian块的正定性来确定。然后对所选子集进行优化，以生成具有最小修改的高影响力对抗示例。大量实验表明，SCP的攻击成功率达到了100%，超越了最先进的稀疏攻击，并以更少的修改为密集攻击提供了卓越的不可感知性。



## **8. Calibrating Uncertainty for Zero-Shot Adversarial CLIP**

校准零镜头对抗CLIP的不确定性 cs.CV

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12997v1) [paper-pdf](https://arxiv.org/pdf/2512.12997v1)

**Authors**: Wenjing lu, Zerui Tao, Dongping Zhang, Yuning Qiu, Yang Yang, Qibin Zhao

**Abstract**: CLIP delivers strong zero-shot classification but remains highly vulnerable to adversarial attacks. Previous work of adversarial fine-tuning largely focuses on matching the predicted logits between clean and adversarial examples, which overlooks uncertainty calibration and may degrade the zero-shot generalization. A common expectation in reliable uncertainty estimation is that predictive uncertainty should increase as inputs become more difficult or shift away from the training distribution. However, we frequently observe the opposite in the adversarial setting: perturbations not only degrade accuracy but also suppress uncertainty, leading to severe miscalibration and unreliable over-confidence. This overlooked phenomenon highlights a critical reliability gap beyond robustness. To bridge this gap, we propose a novel adversarial fine-tuning objective for CLIP considering both prediction accuracy and uncertainty alignments. By reparameterizing the output of CLIP as the concentration parameter of a Dirichlet distribution, we propose a unified representation that captures relative semantic structure and the magnitude of predictive confidence. Our objective aligns these distributions holistically under perturbations, moving beyond single-logit anchoring and restoring calibrated uncertainty. Experiments on multiple zero-shot classification benchmarks demonstrate that our approach effectively restores calibrated uncertainty and achieves competitive adversarial robustness while maintaining clean accuracy.

摘要: CLIP提供了强大的零射击分类，但仍然极易受到对抗攻击。之前的对抗性微调工作主要集中在匹配干净和对抗性示例之间的预测逻辑，这忽视了不确定性校准，并可能降低零镜头概括性。可靠不确定性估计的一个常见期望是，随着输入变得更加困难或偏离训练分布，预测不确定性应该增加。然而，我们在对抗环境中经常观察到相反的情况：扰动不仅会降低准确性，还会抑制不确定性，导致严重的校准错误和不可靠的过度自信。这种被忽视的现象凸显了鲁棒性之外的关键可靠性差距。为了弥合这一差距，我们为CLIP提出了一种新颖的对抗性微调目标，同时考虑预测准确性和不确定性对齐。通过将CLIP的输出重新参数化为Dirichlet分布的浓度参数，我们提出了一个统一的表示，可以捕捉相对语义结构和预测置信度的大小。我们的目标在扰动下整体对齐这些分布，超越单logit锚定并恢复校准的不确定性。多个零镜头分类基准的实验表明，我们的方法有效地恢复了校准的不确定性，并在保持清晰准确性的同时实现了竞争对手鲁棒性。



## **9. Cisco Integrated AI Security and Safety Framework Report**

思科集成人工智能安全和安全框架报告 cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12921v1) [paper-pdf](https://arxiv.org/pdf/2512.12921v1)

**Authors**: Amy Chang, Tiffany Saade, Sanket Mendapara, Adam Swanda, Ankit Garg

**Abstract**: Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements.

摘要: 人工智能（AI）系统正在被轻松而快速地采用，并日益渗透到关键领域：从消费者平台和企业软件到具有嵌入式代理的网络系统。虽然这释放了人类生产力提高的潜力，但攻击面也相应扩大：威胁现在跨越内容安全故障（例如，有害或欺骗性输出）、模型和数据完整性损害（例如，中毒、供应链篡改）、运行时操纵（例如，及时注射、工具和试剂滥用）和生态系统风险（例如，编排滥用、多代理勾结）。MITRE ATLAS、美国国家标准与技术研究院（NIH）AI 100-2对抗性机器学习（ML）分类法以及OWASP大型语言模型（LLM）和统计性人工智能应用程序十大框架提供了有价值的观点，但每个框架都只涵盖了这个多维空间的一部分。   本文介绍了思科的集成人工智能安全框架（“人工智能安全框架”），这是一个统一的、生命周期感知的分类和操作框架，可用于分类、集成和操作全方位人工智能风险。它集成了人工智能安全和跨模式、代理、管道和更广泛生态系统的人工智能安全。人工智能安全框架旨在实用于威胁识别、红色分组、风险优先级，而且它的范围全面，可以扩展到多模式环境中的新兴部署、人形机器人、可穿戴设备和感官基础设施。我们分析了主流框架中的差距，讨论了框架的设计原则，并演示了分类法如何提供结构来理解现代人工智能系统如何失败、对手如何利用这些失败，以及组织如何在整个人工智能生命周期中构建防御，并随着能力的进步而发展。



## **10. CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs**

CTIGuardian：一个用于缓解微调LLM中隐私泄露的少镜头框架 cs.CR

Accepted at the 18th Cybersecurity Experimentation and Test Workshop (CSET), in conjunction with ACSAC 2025

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12914v1) [paper-pdf](https://arxiv.org/pdf/2512.12914v1)

**Authors**: Shashie Dilhara Batan Arachchige, Benjamin Zi Hao Zhao, Hassan Jameel Asghar, Dinusha Vatsalan, Dali Kaafar

**Abstract**: Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains.

摘要: 大型语言模型（LLM）通常经过微调，以使其通用知识适应特定任务和领域，例如网络威胁情报（RTI）。微调主要通过可能包含敏感信息的专有数据集完成。所有者希望他们的微调模型不会无意中将此信息泄露给潜在敌对的最终用户。使用RTI作为用例，我们证明数据提取攻击可以从RTI报告上的微调模型中恢复敏感信息，强调了缓解的必要性。重新训练完整模型以消除这种泄漏在计算上昂贵且不切实际。我们提出了一种替代方法，我们称之为隐私对齐，其灵感来自LLM中的安全对齐。就像安全对齐通过几个例子教导模型遵守安全约束一样，我们通过少量监督来强制隐私对齐，集成了隐私分类器和隐私编辑器，两者都由相同的底层LLM处理。我们使用GPT-4 o mini和Mistral-7 B Direct模型评估我们的系统（称为CTIGGuardian），并以Presidio（命名实体识别（NER）基线）为基准。结果表明，CTIGuardian比基于NER的模型提供了更好的隐私与公用事业权衡。虽然我们在RTI用例中证明了其有效性，但该框架足够通用，可以适用于其他敏感领域。



## **11. PRIVEE: Privacy-Preserving Vertical Federated Learning Against Feature Inference Attacks**

PRIVEE：保护隐私的垂直联邦学习对抗特征推理攻击 cs.LG

12 pages, 3 figures

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.12840v1) [paper-pdf](https://arxiv.org/pdf/2512.12840v1)

**Authors**: Sindhuja Madabushi, Ahmad Faraz Khan, Haider Ali, Ananthram Swami, Rui Ning, Hongyi Wu, Jin-Hee Cho

**Abstract**: Vertical Federated Learning (VFL) enables collaborative model training across organizations that share common user samples but hold disjoint feature spaces. Despite its potential, VFL is susceptible to feature inference attacks, in which adversarial parties exploit shared confidence scores (i.e., prediction probabilities) during inference to reconstruct private input features of other participants. To counter this threat, we propose PRIVEE (PRIvacy-preserving Vertical fEderated lEarning), a novel defense mechanism named after the French word privée, meaning "private." PRIVEE obfuscates confidence scores while preserving critical properties such as relative ranking and inter-score distances. Rather than exposing raw scores, PRIVEE shares only the transformed representations, mitigating the risk of reconstruction attacks without degrading model prediction accuracy. Extensive experiments show that PRIVEE achieves a threefold improvement in privacy protection compared to state-of-the-art defenses, while preserving full predictive performance against advanced feature inference attacks.

摘要: 垂直联邦学习（VFL）支持跨共享共同用户样本但持有不相交特征空间的组织进行协作模型训练。尽管VFL具有潜力，但它很容易受到特征推断攻击，其中敌对方利用共享的置信度分数（即，预测概率）在推理期间重建其他参与者的私人输入特征。为了应对这一威胁，我们提出了PRIVEE（隐私保护垂直fEderated lEarning），这是一种新型防御机制，以法语单词privée命名，意思是“私人”。“PRIVEE混淆了信心分数，同时保留了相对排名和分数间距离等关键属性。PRIVEE不会暴露原始分数，而是仅共享转换后的表示，从而在不降低模型预测准确性的情况下降低了重建攻击的风险。大量实验表明，与最先进的防御相比，PRIVEE在隐私保护方面提高了三倍，同时保留了针对高级特征推断攻击的全面预测性能。



## **12. GradID: Adversarial Detection via Intrinsic Dimensionality of Gradients**

GradID：通过学生的内在模糊性进行对抗检测 cs.LG

16 pages, 8 figures

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.12827v1) [paper-pdf](https://arxiv.org/pdf/2512.12827v1)

**Authors**: Mohammad Mahdi Razmjoo, Mohammad Mahdi Sharifian, Saeed Bagheri Shouraki

**Abstract**: Despite their remarkable performance, deep neural networks exhibit a critical vulnerability: small, often imperceptible, adversarial perturbations can lead to drastically altered model predictions. Given the stringent reliability demands of applications such as medical diagnosis and autonomous driving, robust detection of such adversarial attacks is paramount. In this paper, we investigate the geometric properties of a model's input loss landscape. We analyze the Intrinsic Dimensionality (ID) of the model's gradient parameters, which quantifies the minimal number of coordinates required to describe the data points on their underlying manifold. We reveal a distinct and consistent difference in the ID for natural and adversarial data, which forms the basis of our proposed detection method. We validate our approach across two distinct operational scenarios. First, in a batch-wise context for identifying malicious data groups, our method demonstrates high efficacy on datasets like MNIST and SVHN. Second, in the critical individual-sample setting, we establish new state-of-the-art results on challenging benchmarks such as CIFAR-10 and MS COCO. Our detector significantly surpasses existing methods against a wide array of attacks, including CW and AutoAttack, achieving detection rates consistently above 92\% on CIFAR-10. The results underscore the robustness of our geometric approach, highlighting that intrinsic dimensionality is a powerful fingerprint for adversarial detection across diverse datasets and attack strategies.

摘要: 尽管深度神经网络表现出色，但其表现出一个关键的脆弱性：微小的、通常难以察觉的对抗性扰动可能会导致模型预测发生巨大变化。鉴于医疗诊断和自动驾驶等应用对可靠性的严格要求，对此类对抗攻击的稳健检测至关重要。在本文中，我们研究了模型输入损失景观的几何属性。我们分析模型梯度参数的内在相似性（ID），它量化了描述其基础流上数据点所需的最小坐标数量。我们揭示了自然数据和对抗数据的ID存在明显且一致的差异，这构成了我们提出的检测方法的基础。我们在两种不同的操作场景中验证了我们的方法。首先，在批量识别恶意数据组的上下文中，我们的方法对MNIST和SVHN等数据集表现出高效。其次，在关键的个人样本环境中，我们在CIFAR-10和MS COCO等具有挑战性的基准上建立了新的最先进结果。我们的检测器在抵御包括CW和AutoAttack在内的各种攻击时显着超越了现有方法，在CIFAR-10上实现了始终高于92%的检测率。结果强调了我们几何方法的稳健性，强调了内在维度是跨不同数据集和攻击策略对抗检测的强大指纹。



## **13. Spectral Sentinel: Scalable Byzantine-Robust Decentralized Federated Learning via Sketched Random Matrix Theory on Blockchain**

光谱哨兵：通过区块链上的草图随机矩阵理论进行可扩展的拜占庭鲁棒分散式联邦学习 cs.LG

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.12617v1) [paper-pdf](https://arxiv.org/pdf/2512.12617v1)

**Authors**: Animesh Mishra

**Abstract**: Decentralized federated learning (DFL) enables collaborative model training without centralized trust, but it remains vulnerable to Byzantine clients that poison gradients under heterogeneous (Non-IID) data. Existing defenses face a scalability trilemma: distance-based filtering (e.g., Krum) can reject legitimate Non-IID updates, geometric-median methods incur prohibitive $O(n^2 d)$ cost, and many certified defenses are evaluated only on models below 100M parameters. We propose Spectral Sentinel, a Byzantine detection and aggregation framework that leverages a random-matrix-theoretic signature: honest Non-IID gradients produce covariance eigenspectra whose bulk follows the Marchenko-Pastur law, while Byzantine perturbations induce detectable tail anomalies. Our algorithm combines Frequent Directions sketching with data-dependent MP tracking, enabling detection on models up to 1.5B parameters using $O(k^2)$ memory with $k \ll d$. Under a $(σ,f)$ threat model with coordinate-wise honest variance bounded by $σ^2$ and $f < 1/2$ adversaries, we prove $(ε,δ)$-Byzantine resilience with convergence rate $O(σf / \sqrt{T} + f^2 / T)$, and we provide a matching information-theoretic lower bound $Ω(σf / \sqrt{T})$, establishing minimax optimality. We implement the full system with blockchain integration on Polygon networks and validate it across 144 attack-aggregator configurations, achieving 78.4 percent average accuracy versus 48-63 percent for baseline methods.

摘要: 分散式联邦学习（DFL）支持在没有集中式信任的情况下进行协作模型训练，但它仍然容易受到拜占庭客户端的攻击，这些客户端会在异构（非IID）数据下毒害梯度。现有的防御面临一个可扩展性的三难困境：基于距离的过滤（例如，Krum）可以拒绝合法的非IID更新，几何中位数方法会产生令人望而却步的$O（n ' 2 d）$成本，并且许多经过认证的防御仅在低于1亿个参数的模型上进行评估。我们提出了Spectrum Sentinel，这是一种利用随机矩阵理论签名的拜占庭检测和聚合框架：诚实的非IID梯度产生协方差本征谱，其体积遵循马琴科-帕斯图尔定律，而拜占庭扰动会导致可检测到的尾部异常。我们的算法将频繁方向草图与数据相关的MP跟踪相结合，可以使用$O（k#2）$内存和$k \ll d$来检测最多1.5B参数的模型。在协调诚实方差以$Sigma ' 2 $和$f < 1/2$对手为界的$（Sigma，f）$威胁模型下，我们证明了$（e，δ）$-拜占庭弹性，收敛率为$O（Sigma f / \SQRT{T} + f#39; 2/ T）$，并提供了匹配的信息论下界$Q（Sigma f / \SQRT{T}）$，建立了极小最优性。我们在Polygon网络上实施具有区块链集成的完整系统，并在144种攻击聚合器配置中对其进行验证，平均准确率达到78.4%，而基线方法的平均准确率为48- 63%。



## **14. Unveiling Malicious Logic: Towards a Statement-Level Taxonomy and Dataset for Securing Python Packages**

揭开恶意逻辑：迈向保护Python包的声明级分类和数据集 cs.CR

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.12559v1) [paper-pdf](https://arxiv.org/pdf/2512.12559v1)

**Authors**: Ahmed Ryan, Junaid Mansur Ifti, Md Erfan, Akond Ashfaque Ur Rahman, Md Rayhanur Rahman

**Abstract**: The widespread adoption of open-source ecosystems enables developers to integrate third-party packages, but also exposes them to malicious packages crafted to execute harmful behavior via public repositories such as PyPI. Existing datasets (e.g., pypi-malregistry, DataDog, OpenSSF, MalwareBench) label packages as malicious or benign at the package level, but do not specify which statements implement malicious behavior. This coarse granularity limits research and practice: models cannot be trained to localize malicious code, detectors cannot justify alerts with code-level evidence, and analysts cannot systematically study recurring malicious indicators or attack chains. To address this gap, we construct a statement-level dataset of 370 malicious Python packages (833 files, 90,527 lines) with 2,962 labeled occurrences of malicious indicators. From these annotations, we derive a fine-grained taxonomy of 47 malicious indicators across 7 types that capture how adversarial behavior is implemented in code, and we apply sequential pattern mining to uncover recurring indicator sequences that characterize common attack workflows. Our contribution enables explainable, behavior-centric detection and supports both semantic-aware model training and practical heuristics for strengthening software supply-chain defenses.

摘要: 开源生态系统的广泛采用使开发人员能够集成第三方包，但也会使他们暴露在恶意包中，这些包旨在通过PyPI等公共存储库执行有害行为。现有数据集（例如，pypi-malregistry、DataDog、OpenSSF、MalwareBench）在包级别将包标记为恶意或良性，但没有指定哪些声明实现恶意行为。这种粗糙的粒度限制了研究和实践：无法训练模型来本地化恶意代码，检测器无法用代码级证据证明警报的合理性，分析师无法系统性地研究重复出现的恶意指标或攻击链。为了解决这一差距，我们构建了一个包含370个恶意Python包（833个文件，90，527行）的声明级数据集，其中包含2，962个已标记的恶意指示符。从这些注释中，我们推导出了7种类型的47个恶意指标的细粒度分类法，这些指标捕捉了对抗行为如何在代码中实现，并应用顺序模式挖掘来揭示描述常见攻击工作流程的重复性指标序列。我们的贡献实现了可解释的、以行为为中心的检测，并支持语义感知模型训练和实用启发法，以加强软件供应链防御。



## **15. Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting**

保持灯亮着，保持警惕：能源预测中时间序列LLM的插入式对抗检测 cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12154v1) [paper-pdf](https://arxiv.org/pdf/2512.12154v1)

**Authors**: Hua Ma, Ruoxi Sun, Minhui Xue, Xingliang Yuan, Carsten Rudolph, Surya Nepal, Ling Liu

**Abstract**: Accurate time-series forecasting is increasingly critical for planning and operations in low-carbon power systems. Emerging time-series large language models (TS-LLMs) now deliver this capability at scale, requiring no task-specific retraining, and are quickly becoming essential components within the Internet-of-Energy (IoE) ecosystem. However, their real-world deployment is complicated by a critical vulnerability: adversarial examples (AEs). Detecting these AEs is challenging because (i) adversarial perturbations are optimized across the entire input sequence and exploit global temporal dependencies, which renders local detection methods ineffective, and (ii) unlike traditional forecasting models with fixed input dimensions, TS-LLMs accept sequences of variable length, increasing variability that complicates detection. To address these challenges, we propose a plug-in detection framework that capitalizes on the TS-LLM's own variable-length input capability. Our method uses sampling-induced divergence as a detection signal. Given an input sequence, we generate multiple shortened variants and detect AEs by measuring the consistency of their forecasts: Benign sequences tend to produce stable predictions under sampling, whereas adversarial sequences show low forecast similarity, because perturbations optimized for a full-length sequence do not transfer reliably to shorter, differently-structured subsamples. We evaluate our approach on three representative TS-LLMs (TimeGPT, TimesFM, and TimeLLM) across three energy datasets: ETTh2 (Electricity Transformer Temperature), NI (Hourly Energy Consumption), and Consumption (Hourly Electricity Consumption and Production). Empirical results confirm strong and robust detection performance across both black-box and white-box attack scenarios, highlighting its practicality as a reliable safeguard for TS-LLM forecasting in real-world energy systems.

摘要: 准确的时间序列预测对于低碳电力系统的规划和运营越来越重要。新兴的时间序列大型语言模型（TS-LLM）现在大规模提供了这一能力，无需针对特定任务的再培训，并且正在迅速成为能源互联网（IoE）生态系统中的重要组成部分。然而，它们的现实世界部署因一个关键漏洞而变得复杂：对抗性示例（AE）。检测这些AE具有挑战性，因为（i）对抗性扰动在整个输入序列中得到优化并利用全局时间依赖性，这使得局部检测方法无效，并且（ii）与具有固定输入维度的传统预测模型不同，TS-LLM接受可变长度的序列，增加了使检测复杂化的可变性。为了应对这些挑战，我们提出了一种插件检测框架，该框架利用了TS-LLM自身的可变长度输入能力。我们的方法使用采样引起的分歧作为检测信号。给定一个输入序列，我们生成多个缩短的变体，并通过测量其预测的一致性来检测AE：良性序列往往会在抽样下产生稳定的预测，而对抗序列显示出较低的预测相似性，因为针对全长序列优化的扰动不会可靠地转移到更短、结构不同的子样本。我们在三个能源数据集的三个代表性TS-LLM（TimeGPT、TimesFM和TimeLLM）上评估了我们的方法：ETTh 2（Transformer温度）、NI（小时能源消耗）和消耗（小时电力消耗和生产）。经验结果证实了在黑匣子和白盒攻击场景中强大且稳健的检测性能，凸显了其作为现实世界能源系统中TS-LLM预测的可靠保障的实用性。



## **16. BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry~5.0 Cyber-Physical Systems**

BRIDG-ICS：工业~5.0网络物理系统中智能威胁分析的基于人工智能的知识图 cs.CR

44 Pages, To be published in Springer Cybersecurity Journal

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12112v1) [paper-pdf](https://arxiv.org/pdf/2512.12112v1)

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Abstract**: Industry 5.0's increasing integration of IT and OT systems is transforming industrial operations but also expanding the cyber-physical attack surface. Industrial Control Systems (ICS) face escalating security challenges as traditional siloed defences fail to provide coherent, cross-domain threat insights. We present BRIDG-ICS (BRIDge for Industrial Control Systems), an AI-driven Knowledge Graph (KG) framework for context-aware threat analysis and quantitative assessment of cyber resilience in smart manufacturing environments. BRIDG-ICS fuses heterogeneous industrial and cybersecurity data into an integrated Industrial Security Knowledge Graph linking assets, vulnerabilities, and adversarial behaviours with probabilistic risk metrics (e.g. exploit likelihood, attack cost). This unified graph representation enables multi-stage attack path simulation using graph-analytic techniques. To enrich the graph's semantic depth, the framework leverages Large Language Models (LLMs): domain-specific LLMs extract cybersecurity entities, predict relationships, and translate natural-language threat descriptions into structured graph triples, thereby populating the knowledge graph with missing associations and latent risk indicators. This unified AI-enriched KG supports multi-hop, causality-aware threat reasoning, improving visibility into complex attack chains and guiding data-driven mitigation. In simulated industrial scenarios, BRIDG-ICS scales well, reduces potential attack exposure, and can enhance cyber-physical system resilience in Industry 5.0 settings.

摘要: 工业5.0对IT和OT系统的日益整合正在改变工业运营，但也扩大了网络物理攻击面。工业控制系统（ICS）面临着不断升级的安全挑战，因为传统的孤立防御无法提供连贯的跨域威胁洞察。我们提出了BRIDG-ICS（BRIDge for Industrial Control Systems），这是一个人工智能驱动的知识图（KG）框架，用于智能制造环境中的上下文感知威胁分析和网络弹性的定量评估。BRIDG-ICS将异构的工业和网络安全数据融合到一个集成的工业安全知识图中，将资产、漏洞和对抗行为与概率风险指标（例如，利用可能性、攻击成本）联系起来。这种统一的图形表示可以使用图形分析技术进行多阶段攻击路径模拟。为了丰富图形的语义深度，该框架利用大型语言模型（LLM）：特定领域的LLM提取网络安全实体、预测关系，并将自然语言威胁描述翻译为结构化图形三重体，从而用缺失的关联和潜在风险指标填充知识图形。这款统一的、富含人工智能的KG支持多跳、疏忽感知的威胁推理，提高对复杂攻击链的可见性并指导数据驱动的缓解。在模拟工业场景中，BRIDG-ICS可扩展性良好，减少了潜在的攻击暴露，并可以增强工业5.0环境中的网络物理系统弹性。



## **17. CLOAK: Contrastive Guidance for Latent Diffusion-Based Data Obfuscation**

COTEK：基于潜在扩散的数据混淆的对比指南 cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12086v1) [paper-pdf](https://arxiv.org/pdf/2512.12086v1)

**Authors**: Xin Yang, Omid Ardakanian

**Abstract**: Data obfuscation is a promising technique for mitigating attribute inference attacks by semi-trusted parties with access to time-series data emitted by sensors. Recent advances leverage conditional generative models together with adversarial training or mutual information-based regularization to balance data privacy and utility. However, these methods often require modifying the downstream task, struggle to achieve a satisfactory privacy-utility trade-off, or are computationally intensive, making them impractical for deployment on resource-constrained mobile IoT devices. We propose Cloak, a novel data obfuscation framework based on latent diffusion models. In contrast to prior work, we employ contrastive learning to extract disentangled representations, which guide the latent diffusion process to retain useful information while concealing private information. This approach enables users with diverse privacy needs to navigate the privacy-utility trade-off with minimal retraining. Extensive experiments on four public time-series datasets, spanning multiple sensing modalities, and a dataset of facial images demonstrate that Cloak consistently outperforms state-of-the-art obfuscation techniques and is well-suited for deployment in resource-constrained settings.

摘要: 数据混淆是一种有前途的技术，可以减轻半可信方访问传感器发出的时间序列数据的属性推断攻击。最近的进展利用条件生成模型与对抗训练或基于相互信息的正规化一起来平衡数据隐私和实用性。然而，这些方法通常需要修改下游任务，难以实现令人满意的隐私与公用事业权衡，或者计算密集型，使得它们不适合在资源有限的移动物联网设备上部署。我们提出了Cloak，这是一种基于潜在扩散模型的新型数据混淆框架。与之前的工作相比，我们采用对比学习来提取解开的表示，这引导潜在扩散过程保留有用信息，同时隐藏私人信息。这种方法使具有不同隐私需求的用户能够通过最少的再培训来应对隐私与公用事业的权衡。对跨越多种传感模式的四个公共时间序列数据集和面部图像数据集的广泛实验表明，Cloak始终优于最先进的模糊技术，并且非常适合在资源有限的环境中部署。



## **18. Adversarial Attacks Against Deep Learning-Based Radio Frequency Fingerprint Identification**

针对基于深度学习的射频指纹识别的对抗攻击 cs.CR

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12002v1) [paper-pdf](https://arxiv.org/pdf/2512.12002v1)

**Authors**: Jie Ma, Junqing Zhang, Guanxiong Shen, Alan Marshall, Chip-Hong Chang

**Abstract**: Radio frequency fingerprint identification (RFFI) is an emerging technique for the lightweight authentication of wireless Internet of things (IoT) devices. RFFI exploits deep learning models to extract hardware impairments to uniquely identify wireless devices. Recent studies show deep learning-based RFFI is vulnerable to adversarial attacks. However, effective adversarial attacks against different types of RFFI classifiers have not yet been explored. In this paper, we carried out a comprehensive investigations into different adversarial attack methods on RFFI systems using various deep learning models. Three specific algorithms, fast gradient sign method (FGSM), projected gradient descent (PGD), and universal adversarial perturbation (UAP), were analyzed. The attacks were launched to LoRa-RFFI and the experimental results showed the generated perturbations were effective against convolutional neural networks (CNNs), long short-term memory (LSTM) networks, and gated recurrent units (GRU). We further used UAP to launch practical attacks. Special factors were considered for the wireless context, including implementing real-time attacks, the effectiveness of the attacks over a period of time, etc. Our experimental evaluation demonstrated that UAP can successfully launch adversarial attacks against the RFFI, achieving a success rate of 81.7% when the adversary almost has no prior knowledge of the victim RFFI systems.

摘要: 射频指纹识别（RFFI）是一种用于无线物联网（IOT）设备轻量级认证的新兴技术。RFFI利用深度学习模型来提取硬件损伤，以唯一地识别无线设备。最近的研究表明，基于深度学习的RFFI很容易受到对抗攻击。然而，针对不同类型的RFFI分类器的有效对抗攻击尚未被探索出来。在本文中，我们使用各种深度学习模型对RFFI系统上的不同对抗攻击方法进行了全面研究。分析了三种具体算法，即快速梯度符号法（FGSM）、投影梯度下降法（PVD）和通用对抗扰动法（UAP）。攻击是针对LoRa-RFFI的，实验结果表明生成的扰动对卷积神经网络（CNN）、长短期记忆（LSTM）网络和门控循环单元（GRU）有效。我们进一步使用UAP发起实际攻击。针对无线环境考虑了特殊因素，包括实施实时攻击、攻击在一段时间内的有效性等。我们的实验评估表明，UAP可以成功地对RFFI发起对抗攻击，当对手几乎不了解受害者RFFI系统时，成功率为81.7%。



## **19. Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously**

超级后缀：同时简化文本生成对齐和保护模型 cs.CR

13 pages, 5 Figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11783v1) [paper-pdf](https://arxiv.org/pdf/2512.11783v1)

**Authors**: Andrew Adiletta, Kathryn Adiletta, Kemal Derya, Berk Sunar

**Abstract**: The rapid deployment of Large Language Models (LLMs) has created an urgent need for enhanced security and privacy measures in Machine Learning (ML). LLMs are increasingly being used to process untrusted text inputs and even generate executable code, often while having access to sensitive system controls. To address these security concerns, several companies have introduced guard models, which are smaller, specialized models designed to protect text generation models from adversarial or malicious inputs. In this work, we advance the study of adversarial inputs by introducing Super Suffixes, suffixes capable of overriding multiple alignment objectives across various models with different tokenization schemes. We demonstrate their effectiveness, along with our joint optimization technique, by successfully bypassing the protection mechanisms of Llama Prompt Guard 2 on five different text generation models for malicious text and code generation. To the best of our knowledge, this is the first work to reveal that Llama Prompt Guard 2 can be compromised through joint optimization.   Additionally, by analyzing the changing similarity of a model's internal state to specific concept directions during token sequence processing, we propose an effective and lightweight method to detect Super Suffix attacks. We show that the cosine similarity between the residual stream and certain concept directions serves as a distinctive fingerprint of model intent. Our proposed countermeasure, DeltaGuard, significantly improves the detection of malicious prompts generated through Super Suffixes. It increases the non-benign classification rate to nearly 100%, making DeltaGuard a valuable addition to the guard model stack and enhancing robustness against adversarial prompt attacks.

摘要: 大型语言模型（LLM）的快速部署迫切需要在机器学习（ML）中增强安全和隐私措施。LLM越来越多地被用于处理不受信任的文本输入，甚至生成可执行代码，通常是在可以访问敏感系统控制的情况下。为了解决这些安全问题，几家公司引入了防护模型，这是一种更小的专门模型，旨在保护文本生成模型免受对抗性或恶意输入的影响。在这项工作中，我们通过引入超级后缀来推进对抗性输入的研究，超级后缀能够覆盖具有不同标记化方案的各种模型中的多个对齐目标。我们通过在恶意文本和代码生成的五种不同文本生成模型上成功绕过Llama Promise Guard 2的保护机制，证明了它们以及我们的联合优化技术的有效性。据我们所知，这是第一部揭示Llama Promise Guard 2可以通过联合优化而受到损害的作品。   此外，通过分析令牌序列处理期间模型内部状态与特定概念方向的相似性变化，我们提出了一种有效且轻量级的方法来检测超级后缀攻击。我们表明，剩余流和某些概念方向之间的cos相似性可以作为模型意图的独特指纹。我们提出的对策Delta Guard显着改进了对通过超级后缀生成的恶意提示的检测。它将非良性分类率提高到近100%，使Delta Guard成为保护模型堆栈的宝贵补充，并增强了针对对抗提示攻击的鲁棒性。



## **20. Smudged Fingerprints: A Systematic Evaluation of the Robustness of AI Image Fingerprints**

模糊的指纹：人工智能图像指纹稳健性的系统评估 cs.CV

This work has been accepted for publication in the 4th IEEE Conference on Secure and Trustworthy Machine Learning (IEEE SaTML 2026). The final version will be available on IEEE Xplore

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11771v1) [paper-pdf](https://arxiv.org/pdf/2512.11771v1)

**Authors**: Kai Yao, Marc Juarez

**Abstract**: Model fingerprint detection techniques have emerged as a promising approach for attributing AI-generated images to their source models, but their robustness under adversarial conditions remains largely unexplored. We present the first systematic security evaluation of these techniques, formalizing threat models that encompass both white- and black-box access and two attack goals: fingerprint removal, which erases identifying traces to evade attribution, and fingerprint forgery, which seeks to cause misattribution to a target model. We implement five attack strategies and evaluate 14 representative fingerprinting methods across RGB, frequency, and learned-feature domains on 12 state-of-the-art image generators. Our experiments reveal a pronounced gap between clean and adversarial performance. Removal attacks are highly effective, often achieving success rates above 80% in white-box settings and over 50% under constrained black-box access. While forgery is more challenging than removal, its success significantly varies across targeted models. We also identify a utility-robustness trade-off: methods with the highest attribution accuracy are often vulnerable to attacks. Although some techniques exhibit robustness in specific settings, none achieves high robustness and accuracy across all evaluated threat models. These findings highlight the need for techniques balancing robustness and accuracy, and identify the most promising approaches for advancing this goal.

摘要: 模型指纹检测技术已成为将人工智能生成的图像归因于其源模型的一种有前途的方法，但其在对抗条件下的鲁棒性在很大程度上仍未得到探索。我们对这些技术进行了第一次系统性安全评估，正式化了涵盖白盒和黑盒访问以及两个攻击目标的威胁模型：指纹删除（擦除识别痕迹以逃避归因）和指纹伪造（试图导致错误归因到目标模型）。我们在12个最先进的图像生成器上实施了五种攻击策略，并评估了14种跨越Ruby、频率和学习特征域的代表性指纹识别方法。我们的实验揭示了干净表现和对抗表现之间存在明显差距。删除攻击非常有效，通常在白盒设置中的成功率超过80%，在受限制的黑盒访问下的成功率超过50%。虽然伪造比删除更具挑战性，但其成功在不同目标模型中存在显着差异。我们还确定了实用性与稳健性的权衡：具有最高归因准确性的方法通常容易受到攻击。尽管有些技术在特定设置中表现出鲁棒性，但没有一种技术在所有评估的威胁模型中实现了高鲁棒性和准确性。这些发现凸显了平衡稳健性和准确性的技术的必要性，并确定了实现这一目标的最有希望的方法。



## **21. SpectralKrum: A Spectral-Geometric Defense Against Byzantine Attacks in Federated Learning**

SpectralKrum：联邦学习中针对拜占庭攻击的光谱几何防御 cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11760v1) [paper-pdf](https://arxiv.org/pdf/2512.11760v1)

**Authors**: Aditya Tripathi, Karan Sharma, Rahul Mishra, Tapas Kumar Maiti

**Abstract**: Federated Learning (FL) distributes model training across clients who retain their data locally, but this architecture exposes a fundamental vulnerability: Byzantine clients can inject arbitrarily corrupted updates that degrade or subvert the global model. While robust aggregation methods (including Krum, Bulyan, and coordinate-wise defenses) offer theoretical guarantees under idealized assumptions, their effectiveness erodes substantially when client data distributions are heterogeneous (non-IID) and adversaries can observe or approximate the defense mechanism.   This paper introduces SpectralKrum, a defense that fuses spectral subspace estimation with geometric neighbor-based selection. The core insight is that benign optimization trajectories, despite per-client heterogeneity, concentrate near a low-dimensional manifold that can be estimated from historical aggregates. SpectralKrum projects incoming updates into this learned subspace, applies Krum selection in compressed coordinates, and filters candidates whose orthogonal residual energy exceeds a data-driven threshold. The method requires no auxiliary data, operates entirely on model updates, and preserves FL privacy properties.   We evaluate SpectralKrum against eight robust baselines across seven attack scenarios on CIFAR-10 with Dirichlet-distributed non-IID partitions (alpha = 0.1). Experiments spanning over 56,000 training rounds show that SpectralKrum is competitive against directional and subspace-aware attacks (adaptive-steer, buffer-drift), but offers limited advantage under label-flip and min-max attacks where malicious updates remain spectrally indistinguishable from benign ones.

摘要: 联合学习（FL）在本地保留数据的客户端之间分发模型训练，但这种架构暴露了一个根本性漏洞：拜占庭客户端可以注入任意损坏的更新，从而降级或颠覆全球模型。虽然稳健的聚合方法（包括Krum、Bulyan和协调防御）在理想化假设下提供了理论保证，但当客户数据分布是异类（非IID）并且对手可以观察或逼近防御机制时，它们的有效性会大幅下降。   本文介绍了SpectralKrum，这是一种将谱子空间估计与基于几何邻居的选择相融合的防御方法。核心见解是，尽管每个客户存在差异，但良性优化轨迹仍集中在可以从历史总量估计的低维集合附近。SpectralKrum将输入的更新投影到这个学习的子空间中，在压缩坐标中应用Krum选择，并过滤其垂直剩余能量超过数据驱动阈值的候选。该方法不需要辅助数据，完全根据模型更新操作，并保留FL隐私属性。   我们针对CIFAR-10上七种攻击场景的八个稳健基线来评估SpectralKrum，其中包含Dirichlet分布的非IID分区（Alpha = 0.1）。跨越超过56，000个训练轮的实验表明，SpectralKrum在对抗方向性和子空间感知攻击（自适应转向、缓冲区漂移）时具有竞争力，但在标签翻转和最小最大攻击下提供的优势有限，恶意更新在频谱上与良性更新无法区分。



## **22. CLINIC: Evaluating Multilingual Trustworthiness in Language Models for Healthcare**

评估医疗保健语言模型中的多语言可信度 cs.CL

49 pages, 31 figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11437v1) [paper-pdf](https://arxiv.org/pdf/2512.11437v1)

**Authors**: Akash Ghosh, Srivarshinee Sridhar, Raghav Kaushik Ravi, Muhsin Muhsin, Sriparna Saha, Chirag Agarwal

**Abstract**: Integrating language models (LMs) in healthcare systems holds great promise for improving medical workflows and decision-making. However, a critical barrier to their real-world adoption is the lack of reliable evaluation of their trustworthiness, especially in multilingual healthcare settings. Existing LMs are predominantly trained in high-resource languages, making them ill-equipped to handle the complexity and diversity of healthcare queries in mid- and low-resource languages, posing significant challenges for deploying them in global healthcare contexts where linguistic diversity is key. In this work, we present CLINIC, a Comprehensive Multilingual Benchmark to evaluate the trustworthiness of language models in healthcare. CLINIC systematically benchmarks LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy, operationalized through 18 diverse tasks, spanning 15 languages (covering all the major continents), and encompassing a wide array of critical healthcare topics like disease conditions, preventive actions, diagnostic tests, treatments, surgeries, and medications. Our extensive evaluation reveals that LMs struggle with factual correctness, demonstrate bias across demographic and linguistic groups, and are susceptible to privacy breaches and adversarial attacks. By highlighting these shortcomings, CLINIC lays the foundation for enhancing the global reach and safety of LMs in healthcare across diverse languages.

摘要: 将语言模型（LM）集成到医疗保健系统中对于改善医疗工作流程和决策具有巨大的前景。然而，现实世界采用它们的一个关键障碍是缺乏对其可信度的可靠评估，尤其是在多语言医疗保健环境中。现有的LM主要接受高资源语言培训，这使得它们无法处理中低资源语言中的医疗保健查询的复杂性和多样性，这对在语言多样性至关重要的全球医疗保健环境中部署它们构成了重大挑战。在这项工作中，我们介绍了CLARIC，这是一个全面的多语言基准，用于评估医疗保健中语言模型的可信度。CLARIC在可信度的五个关键方面对LM进行了系统性的基准测试：真实性、公平性、安全性、稳健性和隐私，通过18项不同的任务来实施，跨越15种语言（覆盖所有主要大陆），并涵盖各种关键的医疗保健主题，例如疾病状况、预防行动、诊断测试、治疗、手术和药物。我们的广泛评估表明，LM在事实正确性方面遇到困难，表现出跨人口和语言群体的偏见，并且容易受到隐私侵犯和对抗性攻击。通过强调这些缺点，CLARIC为增强LM在不同语言的医疗保健领域的全球影响力和安全性奠定了基础。



## **23. Attacking and Securing Community Detection: A Game-Theoretic Framework**

攻击和保护社区检测：游戏理论框架 cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11359v1) [paper-pdf](https://arxiv.org/pdf/2512.11359v1)

**Authors**: Yifan Niu, Aochuan Chen, Tingyang Xu, Jia Li

**Abstract**: It has been demonstrated that adversarial graphs, i.e., graphs with imperceptible perturbations, can cause deep graph models to fail on classification tasks. In this work, we extend the concept of adversarial graphs to the community detection problem, which is more challenging. We propose novel attack and defense techniques for community detection problem, with the objective of hiding targeted individuals from detection models and enhancing the robustness of community detection models, respectively. These techniques have many applications in real-world scenarios, for example, protecting personal privacy in social networks and understanding camouflage patterns in transaction networks. To simulate interactive attack and defense behaviors, we further propose a game-theoretic framework, called CD-GAME. One player is a graph attacker, while the other player is a Rayleigh Quotient defender. The CD-GAME models the mutual influence and feedback mechanisms between the attacker and the defender, revealing the dynamic evolutionary process of the game. Both players dynamically update their strategies until they reach the Nash equilibrium. Extensive experiments demonstrate the effectiveness of our proposed attack and defense methods, and both outperform existing baselines by a significant margin. Furthermore, CD-GAME provides valuable insights for understanding interactive attack and defense scenarios in community detection problems. We found that in traditional single-step attack or defense, attacker tends to employ strategies that are most effective, but are easily detected and countered by defender. When the interactive game reaches a Nash equilibrium, attacker adopts more imperceptible strategies that can still achieve satisfactory attack effectiveness even after defense.

摘要: 已经证明，对抗图，即，具有不可感知的扰动的图可能会导致深度图模型在分类任务中失败。在这项工作中，我们将对抗图的概念扩展到更具挑战性的社区检测问题。我们针对社区检测问题提出了新颖的攻击和防御技术，目标分别是将目标个体隐藏在检测模型中并增强社区检测模型的鲁棒性。这些技术在现实世界场景中有许多应用，例如保护社交网络中的个人隐私和理解交易网络中的伪装模式。为了模拟交互式攻击和防御行为，我们进一步提出了一个博弈论框架，称为CD-GAME。一名玩家是图形攻击者，而另一名玩家是Rayleigh商防御者。CD-GAME模拟了攻击者和防御者之间的相互影响和反馈机制，揭示了游戏的动态进化过程。两个参与者都动态更新他们的策略，直到达到纳什均衡。大量的实验证明了我们提出的攻击和防御方法的有效性，并且两者的性能都远远优于现有的基线。此外，CD-GAME为理解社区检测问题中的交互式攻击和防御场景提供了宝贵的见解。我们发现，在传统的一步攻击或防御中，攻击者倾向于采用最有效但易于被防御者检测和反击的策略。当交互式游戏达到纳什均衡时，攻击者会采取更加不可感知的策略，即使在防守后仍然可以达到令人满意的攻击效果。



## **24. Empirical evaluation of the Frank-Wolfe methods for constructing white-box adversarial attacks**

构建白盒对抗攻击的Frank-Wolfe方法的经验评估 cs.LG

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10936v1) [paper-pdf](https://arxiv.org/pdf/2512.10936v1)

**Authors**: Kristina Korotkova, Aleksandr Katrutsa

**Abstract**: The construction of adversarial attacks for neural networks appears to be a crucial challenge for their deployment in various services. To estimate the adversarial robustness of a neural network, a fast and efficient approach is needed to construct adversarial attacks. Since the formalization of adversarial attack construction involves solving a specific optimization problem, we consider the problem of constructing an efficient and effective adversarial attack from a numerical optimization perspective. Specifically, we suggest utilizing advanced projection-free methods, known as modified Frank-Wolfe methods, to construct white-box adversarial attacks on the given input data. We perform a theoretical and numerical evaluation of these methods and compare them with standard approaches based on projection operations or geometrical intuition. Numerical experiments are performed on the MNIST and CIFAR-10 datasets, utilizing a multiclass logistic regression model, the convolutional neural networks (CNNs), and the Vision Transformer (ViT).

摘要: 神经网络的对抗攻击的构建似乎是其在各种服务中部署的一个关键挑战。为了估计神经网络的对抗鲁棒性，需要一种快速有效的方法来构建对抗攻击。由于对抗攻击构建的形式化涉及解决特定的优化问题，因此我们从数字优化的角度考虑构建高效且有效的对抗攻击的问题。具体来说，我们建议利用先进的无投影方法（称为修改的Frank-Wolfe方法）来对给定输入数据构建白盒对抗攻击。我们对这些方法进行理论和数值评估，并将它们与基于投影操作或几何直觉的标准方法进行比较。在MNIST和CIFAR-10数据集上进行数值实验，利用多类逻辑回归模型，卷积神经网络（CNN）和视觉Transformer（ViT）。



## **25. Adaptive Intrusion Detection System Leveraging Dynamic Neural Models with Adversarial Learning for 5G/6G Networks**

利用动态神经模型和对抗学习的5G/6 G网络自适应入侵检测系统 cs.CR

7 pages,3 figures, 2 Table. Neha and T. Bhatia "Adaptive Intrusion Detection System Leveraging Dynamic Neural Models with Adversarial Learning for 5G/6G Networks" (2025) 103-107

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.10637v2) [paper-pdf](https://arxiv.org/pdf/2512.10637v2)

**Authors**: Neha, Tarunpreet Bhatia

**Abstract**: Intrusion Detection Systems (IDS) are critical components in safeguarding 5G/6G networks from both internal and external cyber threats. While traditional IDS approaches rely heavily on signature-based methods, they struggle to detect novel and evolving attacks. This paper presents an advanced IDS framework that leverages adversarial training and dynamic neural networks in 5G/6G networks to enhance network security by providing robust, real-time threat detection and response capabilities. Unlike conventional models, which require costly retraining to update knowledge, the proposed framework integrates incremental learning algorithms, reducing the need for frequent retraining. Adversarial training is used to fortify the IDS against poisoned data. By using fewer features and incorporating statistical properties, the system can efficiently detect potential threats. Extensive evaluations using the NSL- KDD dataset demonstrate that the proposed approach provides better accuracy of 82.33% for multiclass classification of various network attacks while resisting dataset poisoning. This research highlights the potential of adversarial-trained, dynamic neural networks for building resilient IDS solutions.

摘要: 入侵检测系统（IDS）是保护5G/6 G网络免受内部和外部网络威胁的关键组件。虽然传统的IDS方法严重依赖基于签名的方法，但它们很难检测新颖且不断发展的攻击。本文提出了一种先进的IDS框架，该框架利用5G/6 G网络中的对抗性训练和动态神经网络，通过提供稳健、实时的威胁检测和响应能力来增强网络安全性。与需要昂贵的重新训练来更新知识的传统模型不同，拟议的框架集成了增量学习算法，减少了频繁重新训练的需要。对抗训练用于加强IDS免受有毒数据的侵害。通过使用更少的功能并结合统计属性，该系统可以有效地检测潜在威胁。使用NSL- KDD数据集进行的广泛评估表明，所提出的方法为各种网络攻击的多类分类提供了82.33%的更好准确性，同时可以抵抗数据集中毒。这项研究强调了经过对抗训练的动态神经网络在构建弹性IDS解决方案方面的潜力。



## **26. Authority Backdoor: A Certifiable Backdoor Mechanism for Authoring DNNs**

权威后门：一种用于编写DNN的可认证后门机制 cs.CR

Accepted to AAAI 2026 (Main Track). Code is available at: https://github.com/PlayerYangh/Authority-Trigger

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10600v1) [paper-pdf](https://arxiv.org/pdf/2512.10600v1)

**Authors**: Han Yang, Shaofeng Li, Tian Dong, Xiangyu Xu, Guangchi Liu, Zhen Ling

**Abstract**: Deep Neural Networks (DNNs), as valuable intellectual property, face unauthorized use. Existing protections, such as digital watermarking, are largely passive; they provide only post-hoc ownership verification and cannot actively prevent the illicit use of a stolen model. This work proposes a proactive protection scheme, dubbed ``Authority Backdoor," which embeds access constraints directly into the model. In particular, the scheme utilizes a backdoor learning framework to intrinsically lock a model's utility, such that it performs normally only in the presence of a specific trigger (e.g., a hardware fingerprint). But in its absence, the DNN's performance degrades to be useless. To further enhance the security of the proposed authority scheme, the certifiable robustness is integrated to prevent an adaptive attacker from removing the implanted backdoor. The resulting framework establishes a secure authority mechanism for DNNs, combining access control with certifiable robustness against adversarial attacks. Extensive experiments on diverse architectures and datasets validate the effectiveness and certifiable robustness of the proposed framework.

摘要: 深度神经网络（DNN）作为宝贵的知识产权，面临未经授权的使用。现有的保护措施（例如数字水印）在很大程度上是被动的;它们仅提供事后所有权验证，无法积极防止非法使用被盗模型。这项工作提出了一种被称为“权威后门”的主动保护计划，该计划将访问限制直接嵌入到模型中。特别是，该方案利用后门学习框架来本质上锁定模型的效用，以便它仅在存在特定触发器（例如，硬件指纹）。但如果没有它，DNN的性能就会下降到无用。为了进一步增强拟议授权方案的安全性，集成了可认证的鲁棒性，以防止自适应攻击者删除植入的后门。由此产生的框架为DNN建立了安全的授权机制，将访问控制与针对对抗攻击的可认证鲁棒性相结合。对不同架构和数据集的广泛实验验证了拟议框架的有效性和可认证的稳健性。



## **27. T-ADD: Enhancing DOA Estimation Robustness Against Adversarial Attacks**

T-ADD：增强DOE估计针对对抗攻击的鲁棒性 eess.SP

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10496v1) [paper-pdf](https://arxiv.org/pdf/2512.10496v1)

**Authors**: Shilian Zheng, Xiaoxiang Wu, Luxin Zhang, Keqiang Yue, Peihan Qi, Zhijin Zhao

**Abstract**: Deep learning has achieved remarkable success in direction-of-arrival (DOA) estimation. However, recent studies have shown that adversarial perturbations can severely compromise the performance of such models. To address this vulnerability, we propose Transformer-based Adversarial Defense for DOA estimation (T-ADD), a transformer-based defense method designed to counter adversarial attacks. To achieve a balance between robustness and estimation accuracy, we formulate the adversarial defense as a joint reconstruction task and introduce a tailored joint loss function. Experimental results demonstrate that, compared with three state-of-the-art adversarial defense methods, the proposed T-ADD significantly mitigates the adverse effects of widely used adversarial attacks, leading to notable improvements in the adversarial robustness of the DOA model.

摘要: 深度学习在到达方向（DOE）估计方面取得了显着成功。然而，最近的研究表明，对抗性扰动可能会严重损害此类模型的性能。为了解决这个漏洞，我们提出了基于变换器的针对到达目的地估计的对抗性防御（T-ADD），这是一种基于变换器的防御方法，旨在对抗对抗性攻击。为了实现稳健性和估计准确性之间的平衡，我们将对抗防御制定为联合重建任务，并引入了定制的联合损失函数。实验结果表明，与三种最先进的对抗性防御方法相比，所提出的T-ADD显着减轻了广泛使用的对抗性攻击的不利影响，从而使DOE模型的对抗性鲁棒性显着提高。



## **28. Stealth and Evasion in Rogue AP Attacks: An Analysis of Modern Detection and Bypass Techniques**

流氓AP攻击中的隐身与规避：现代检测与绕过技术分析 cs.CR

5 pages, 3 figures, experimental paper

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10470v1) [paper-pdf](https://arxiv.org/pdf/2512.10470v1)

**Authors**: Kaleb Bacztub, Braden Vester, Matteo Hodge, Liulseged Abate

**Abstract**: Wireless networks act as the backbone of modern digital connectivity, making them a primary target for cyber adversaries. Rogue Access Point attacks, specifically the Evil Twin variant, enable attackers to clone legitimate wireless network identifiers to deceive users into connecting. Once a connection is established, the adversary can intercept traffic and harvest sensitive credentials. While modern defensive architectures often employ Network Intrusion Detection Systems (NIDS) to identify malicious activity, the effectiveness of these systems against Layer 2 wireless threats remains a subject of critical inquiry. This project aimed to design a stealth-capable Rogue AP and evaluate its detectability against Suricata, an open-source NIDS/IPS. The methodology initially focused on a hardware-based deployment using Raspberry Pi platforms but transitioned to a virtualized environment due to severe system compatibility issues. Using Wifipumpkin3, the research team successfully deployed a captive portal that harvested user credentials from connected devices. However, the Suricata NIDS failed to flag the attack, highlighting a significant blind spot in traditional intrusion detection regarding wireless management frame attacks. This paper details the construction of the attack, the evasion techniques employed, and the limitations of current NIDS solutions in detecting localized wireless threats

摘要: 无线网络是现代数字连接的支柱，使其成为网络对手的主要目标。流氓接入点攻击，特别是Evil Twin变体，使攻击者能够克隆合法的无线网络标识符以欺骗用户进行连接。建立连接后，对手可以拦截流量并获取敏感凭证。虽然现代防御架构通常使用网络入侵检测系统（NIDS）来识别恶意活动，但这些系统针对第2层无线威胁的有效性仍然是一个重要调查的主题。该项目旨在设计具有隐身能力的Rogue AP，并评估其针对开源NIDS/IPS Suricata的可检测性。该方法最初专注于使用Raspberry Pi平台的基于硬件的部署，但由于严重的系统兼容性问题而过渡到虚拟化环境。使用Wifipumpkin 3，研究团队成功部署了一个专属门户，该门户可以从连接的设备获取用户凭据。然而，Suricata NIDS未能标记该攻击，凸显了传统入侵检测中有关无线管理帧攻击的一个重大盲点。本文详细介绍了攻击的结构、所采用的规避技术以及当前NIDS解决方案在检测局部无线威胁方面的局限性



## **29. When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection**

当批评变成接受：量化基于LLM-based科学评论员间接提示注入的脆弱性 cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.10449v2) [paper-pdf](https://arxiv.org/pdf/2512.10449v2)

**Authors**: Devanshu Sahoo, Manish Prasad, Vasudev Majhi, Jahnvi Singh, Vinay Chamola, Yash Sinha, Murari Mandal, Dhruv Kumar

**Abstract**: The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.

摘要: 随着大型语言模型（LLM）的集成，科学同行评审的格局正在迅速发展。这种转变是由两个平行趋势推动的：评审员广泛采用法学硕士来管理工作量（“懒惰评审员”假设），以及AAAI和斯坦福大学Agents 4Science等会议正式机构部署人工智能驱动的评估系统。本研究调查了这些“法学硕士作为法官”系统（包括非法的和受制裁的）对对抗性PDF操纵的稳健性。与一般的越狱不同，我们专注于一个独特的激励：将“卸载”决策转换为“接受”，为此我们开发了一种新型的评估指标，称为WAVS（加权对抗性脆弱性分数）。我们策划了一个包含200篇科学论文的数据集，并为这项任务调整了15种特定领域的攻击策略，在13种语言模型中进行了评估，包括GPT-5，Claude Haiku和DeepSeek。我们的研究结果表明，混淆策略，如“最大马克Magyk”成功地操纵分数，即使在大规模的模型中也能达到惊人的决策翻转率。我们将发布完整的数据集和注入框架，以促进对该主题的更多研究。



## **30. Differential Privacy for Secure Machine Learning in Healthcare IoT-Cloud Systems**

医疗保健物联网云系统中安全机器学习的差异隐私 cs.CR

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10426v1) [paper-pdf](https://arxiv.org/pdf/2512.10426v1)

**Authors**: N Mangala, Murtaza Rangwala, S Aishwarya, B Eswara Reddy, Rajkumar Buyya, KR Venugopal, SS Iyengar, LM Patnaik

**Abstract**: Healthcare has become exceptionally sophisticated, as wearables and connected medical devices are revolutionising remote patient monitoring, emergency response, medication management, diagnosis, and predictive and prescriptive analytics. Internet of Things and Cloud computing integrated systems (IoT-Cloud) facilitate sensing, automation, and processing for these healthcare applications. While real-time response is crucial for alleviating patient emergencies, protecting patient privacy is extremely important in data-driven healthcare. In this paper, we propose a multi-layer IoT, Edge and Cloud architecture to enhance the speed of response for emergency healthcare by distributing tasks based on response criticality and permanence of storage. Privacy of patient data is assured by proposing a Differential Privacy framework across several machine learning models such as K-means, Logistic Regression, Random Forest and Naive Bayes. We establish a comprehensive threat model identifying three adversary classes and evaluate Laplace, Gaussian, and hybrid noise mechanisms across varying privacy budgets, with supervised algorithms achieving up to 86% accuracy. The proposed hybrid Laplace-Gaussian noise mechanism with adaptive budget allocation provides a balanced approach, offering moderate tails and better privacy-utility trade-offs for both low and high dimension datasets. At the practical threshold of $\varepsilon = 5.0$, supervised algorithms achieve 82-84% accuracy while reducing attribute inference attacks by up to 18% and data reconstruction correlation by 70%. Blockchain security further ensures trusted communication through time-stamping, traceability, and immutability for analytics applications. Edge computing demonstrates 8$\times$ latency reduction for emergency scenarios, validating the hierarchical architecture for time-critical operations.

摘要: 医疗保健已经变得异常复杂，因为可穿戴设备和联网医疗设备正在彻底改变远程患者监控、应急响应、药物管理、诊断以及预测性和规范性分析。物联网和云计算集成系统（IoT-Cloud）促进了这些医疗保健应用的传感、自动化和处理。虽然实时响应对于缓解患者紧急情况至关重要，但保护患者隐私在数据驱动的医疗保健中极其重要。在本文中，我们提出了一种多层物联网、边缘和云架构，通过根据响应关键性和存储持久性分配任务来提高紧急医疗保健的响应速度。通过提出跨K-means、Logical Regulation、Random Forest和Naive Bayes等多个机器学习模型的差异隐私框架来确保患者数据的隐私。我们建立了一个全面的威胁模型，识别三种对手类别，并评估不同隐私预算的拉普拉斯、高斯和混合噪音机制，监督算法的准确率高达86%。提出的具有自适应预算分配的混合拉普拉斯-高斯噪音机制提供了一种平衡的方法，为低维和高维数据集提供适度的尾部和更好的隐私-效用权衡。在$\varepð = 5.0$的实际阈值下，监督算法可实现82-84%的准确性，同时将属性推理攻击减少高达18%，并将数据重建相关性减少70%。区块链安全通过分析应用程序的时间戳、可追溯性和不变性进一步确保可信的通信。边缘计算演示了紧急情况下的延迟可减少8 $\x $，验证了时间紧迫的操作的分层架构。



## **31. Phishing Email Detection Using Large Language Models**

使用大型语言模型的网络钓鱼电子邮件检测 cs.CR

7 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.10104v2) [paper-pdf](https://arxiv.org/pdf/2512.10104v2)

**Authors**: Najmul Hasan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination.

摘要: 电子邮件网络钓鱼是最普遍、最具全球影响力的网络入侵载体之一。随着系统越来越多地部署大型语言模型（LLM）应用程序，这些系统面临着利用其基本架构的不断发展的网络钓鱼电子邮件威胁。当前的LLM在部署到电子邮件安全系统之前需要进行实质性的强化，特别是针对利用架构漏洞的协调多载体攻击。本文提出了LLMPEA，这是一个基于LLM的框架，用于检测跨多种攻击载体的网络钓鱼电子邮件攻击，包括提示注入、文本细化和多语言攻击。我们评估了三个前沿LLM（例如，GPT-4 o、Claude Sonnet 4和Grok-3）以及全面的提示设计，以评估其可行性、稳健性和针对网络钓鱼电子邮件攻击的限制。我们的实证分析表明，LLM可以检测到超过90%的网络钓鱼电子邮件，同时我们还强调，基于LLM的网络钓鱼电子邮件检测系统可能会被对抗性攻击、提示注入和多语言攻击所利用。我们的研究结果为攻击者组合利用多个漏洞的现实环境中基于LLM的网络钓鱼检测提供了重要见解。



## **32. Optimization-Guided Diffusion for Interactive Scene Generation**

交互式场景生成的优化引导扩散 cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.07661v2) [paper-pdf](https://arxiv.org/pdf/2512.07661v2)

**Authors**: Shihao Li, Naisheng Ye, Tianyu Li, Kashyap Chitta, Tuo An, Peng Su, Boyang Wang, Haiou Liu, Chen Lv, Hongyang Li

**Abstract**: Realistic and diverse multi-agent driving scenes are crucial for evaluating autonomous vehicles, but safety-critical events which are essential for this task are rare and underrepresented in driving datasets. Data-driven scene generation offers a low-cost alternative by synthesizing complex traffic behaviors from existing driving logs. However, existing models often lack controllability or yield samples that violate physical or social constraints, limiting their usability. We present OMEGA, an optimization-guided, training-free framework that enforces structural consistency and interaction awareness during diffusion-based sampling from a scene generation model. OMEGA re-anchors each reverse diffusion step via constrained optimization, steering the generation towards physically plausible and behaviorally coherent trajectories. Building on this framework, we formulate ego-attacker interactions as a game-theoretic optimization in the distribution space, approximating Nash equilibria to generate realistic, safety-critical adversarial scenarios. Experiments on nuPlan and Waymo show that OMEGA improves generation realism, consistency, and controllability, increasing the ratio of physically and behaviorally valid scenes from 32.35% to 72.27% for free exploration capabilities, and from 11% to 80% for controllability-focused generation. Our approach can also generate $5\times$ more near-collision frames with a time-to-collision under three seconds while maintaining the overall scene realism.

摘要: 真实且多样化的多智能体驾驶场景对于评估自动驾驶汽车至关重要，但对这项任务至关重要的安全关键事件在驾驶数据集中很少且代表性不足。数据驱动的场景生成通过从现有驾驶日志合成复杂的交通行为来提供低成本的替代方案。然而，现有模型通常缺乏可控制性或产生违反物理或社会约束的样本，从而限制了其可用性。我们提出了OMEK，这是一个优化引导、免训练的框架，它在场景生成模型的基于扩散的采样期间强制执行结构一致性和交互意识。OMEK通过约束优化重新锚定每个反向扩散步骤，引导一代走向物理上合理且行为上一致的轨迹。基于这个框架，我们将自我攻击者的相互作用制定为分布空间中的博弈论优化，逼近纳什均衡以生成现实的、对安全至关重要的对抗场景。在nuPlan和Waymo上的实验表明，OMEK提高了世代的真实感、一致性和可控性，将物理和行为有效场景的比例从32.35%提高到72.27%（对于免费探索能力），从11%提高到80%（对于以可开发性为重点的世代）。我们的方法还可以生成5美元以上的近碰撞帧，碰撞时间在三秒以内，同时保持整体场景的真实感。



## **33. Defense That Attacks: How Robust Models Become Better Attackers**

攻击的防御：稳健模型如何成为更好的攻击者 cs.CV

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.02830v3) [paper-pdf](https://arxiv.org/pdf/2512.02830v3)

**Authors**: Mohamed Awad, Mahmoud Akrm, Walid Gomaa

**Abstract**: Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples.

摘要: 深度学习在计算机视觉领域取得了巨大成功，但仍然容易受到对抗攻击。对抗训练是旨在提高模型稳健性的主要防御措施。然而，它对攻击可转移性的影响尚未得到充分研究。在这项工作中，我们询问对抗性训练是否无意中增加了对抗性示例的可移植性。为了解决这个问题，我们训练了一个由36个模型组成的多样化动物园，包括CNN和ViT，并进行了全面的可移植性实验。我们的结果揭示了一个明显的悖论：对抗训练（AT）模型会产生比标准模型更有效地传递的扰动，从而引入了新的生态系统风险。为了实现可重复性和进一步研究，我们发布了所有模型、代码和实验脚本。此外，我们认为稳健性评估不仅应该评估模型对转移攻击的抵抗力，还应该评估其产生可转移对抗示例的倾向。



## **34. SA$^{2}$GFM: Enhancing Robust Graph Foundation Models with Structure-Aware Semantic Augmentation**

SA$^{2}$GFM：通过结构感知语义增强稳健图基础模型 cs.LG

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.07857v2) [paper-pdf](https://arxiv.org/pdf/2512.07857v2)

**Authors**: Junhua Shi, Qingyun Sun, Haonan Yuan, Xingcheng Fu

**Abstract**: We present Graph Foundation Models (GFMs) which have made significant progress in various tasks, but their robustness against domain noise, structural perturbations, and adversarial attacks remains underexplored. A key limitation is the insufficient modeling of hierarchical structural semantics, which are crucial for generalization. In this paper, we propose SA$^{2}$GFM, a robust GFM framework that improves domain-adaptive representations through Structure-Aware Semantic Augmentation. First, we encode hierarchical structural priors by transforming entropy-based encoding trees into structure-aware textual prompts for feature augmentation. The enhanced inputs are processed by a self-supervised Information Bottleneck mechanism that distills robust, transferable representations via structure-guided compression. To address negative transfer in cross-domain adaptation, we introduce an expert adaptive routing mechanism, combining a mixture-of-experts architecture with a null expert design. For efficient downstream adaptation, we propose a fine-tuning module that optimizes hierarchical structures through joint intra- and inter-community structure learning. Extensive experiments demonstrate that SA$^{2}$GFM outperforms 9 state-of-the-art baselines in terms of effectiveness and robustness against random noise and adversarial perturbations for node and graph classification.

摘要: 我们提出的图形基础模型（GFM）在各种任务中取得了显着进展，但其对域噪音、结构扰动和对抗攻击的鲁棒性仍然没有得到充分的研究。一个关键限制是分层结构语义的建模不足，而分层结构语义对于概括至关重要。在本文中，我们提出SA$^{2}$GFM，这是一个强大的GFM框架，通过结构感知语义增强来改进域自适应表示。首先，我们通过将基于熵的编码树转换为结构感知的文本提示来编码分层结构先验以进行特征增强。增强的输入由自我监督的信息瓶颈机制处理，该机制通过结构引导压缩提取稳健的、可转移的表示。为了解决跨域适应中的负转移问题，我们引入了一种专家自适应路由机制，将混合专家架构与空专家设计相结合。为了高效的下游适应，我们提出了一个微调模块，该模块通过社区内和社区间的联合结构学习来优化分层结构。大量实验表明，SA$^{2}$GFM在针对随机噪音和节点和图形分类对抗性扰动的有效性和鲁棒性方面优于9个最先进的基线。



## **35. ATAC: Augmentation-Based Test-Time Adversarial Correction for CLIP**

ATAC：CLIP的基于增强的测试时对抗纠正 cs.CV

16 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2511.17362v2) [paper-pdf](https://arxiv.org/pdf/2511.17362v2)

**Authors**: Linxiang Su, András Balogh

**Abstract**: Despite its remarkable success in zero-shot image-text matching, CLIP remains highly vulnerable to adversarial perturbations on images. As adversarial fine-tuning is prohibitively costly, recent works explore various test-time defense strategies; however, these approaches still exhibit limited robustness. In this work, we revisit this problem and propose a simple yet effective strategy: Augmentation-based Test-time Adversarial Correction (ATAC). Our method operates directly in the embedding space of CLIP, calculating augmentation-induced drift vectors to infer a semantic recovery direction and correcting the embedding based on the angular consistency of these latent drifts. Across a wide range of benchmarks, ATAC consistently achieves remarkably high robustness, surpassing that of previous state-of-the-art methods by nearly 50\% on average, all while requiring minimal computational overhead. Furthermore, ATAC retains state-of-the-art robustness in unconventional and extreme settings and even achieves nontrivial robustness against adaptive attacks. Our results demonstrate that ATAC is an efficient method in a novel paradigm for test-time adversarial defenses in the embedding space of CLIP.

摘要: 尽管CLIP在零镜头图像文本匹配方面取得了显着成功，但仍然极易受到图像上的对抗性扰动的影响。由于对抗性微调的成本高得令人望而却步，最近的作品探索了各种测试时防御策略;然而，这些方法仍然表现出有限的稳健性。在这项工作中，我们重新审视了这个问题，并提出了一种简单而有效的策略：基于增强的测试时对抗纠正（ATAC）。我们的方法直接在CLIP的嵌入空间中操作，计算增强引起的漂移量以推断语义恢复方向，并基于这些潜在漂移的角度一致性来纠正嵌入。在各种基准测试中，ATAC始终实现了非常高的稳健性，平均比之前最先进的方法高出近50%，同时需要最小的计算负担。此外，ATAC在非常规和极端环境中保留了最先进的鲁棒性，甚至实现了针对自适应攻击的非凡鲁棒性。我们的结果表明，ATAC是CLIP嵌入空间中测试时对抗防御的新型范式中的一种有效方法。



## **36. Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion**

物理真实的序列级对抗服装，用于鲁棒的人体检测逃避 cs.CV

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2511.16020v2) [paper-pdf](https://arxiv.org/pdf/2511.16020v2)

**Authors**: Dingkun Zhou, Patrick P. K. Chan, Hengxu Wu, Shikang Zheng, Ruiqi Huang, Yuanjie Zhao

**Abstract**: Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility.

摘要: 用于人类检测的深度神经网络极易受到对抗性操纵的影响，从而在真实监控环境中造成安全和隐私风险。可穿戴攻击提供了一个现实的威胁模型，但现有的方法通常逐帧优化纹理，因此无法在具有运动、姿势变化和服装变形的长视频序列中保持隐藏。在这项工作中，引入了序列级优化框架，为衬衫、裤子和帽子生成自然、可打印的对抗纹理，这些纹理在数字和物理环境中的整个行走视频中仍然有效。产品图像首先映射到紫外空间，然后转换为紧凑的调色板和控制点参数化，并通过ICC锁定以保持所有颜色可打印。然后使用基于物理的人体-服装管道来模拟运动、多角度摄像机视角、布料动态和照明变化。使用具有时间加权的期望转换目标来优化控制点，以便在整个序列中最小化检测置信度。大量实验证明了强大且稳定的隐藏性、对观点变化的高鲁棒性以及卓越的跨模型可移植性。采用升华印花生产的物理服装在室内和室外记录下实现了可靠的抑制，证实了现实世界的可行性。



## **37. Cyber-Resilient Data-Driven Event-Triggered Secure Control for Autonomous Vehicles Under False Data Injection Attacks**

虚假数据注入攻击下自动驾驶汽车的网络弹性数据驱动事件触发安全控制 eess.SY

14 pages, 8 figures

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2511.15925v2) [paper-pdf](https://arxiv.org/pdf/2511.15925v2)

**Authors**: Yashar Mousavi, Mahsa Tavasoli, Ibrahim Beklan Kucukdemiral, Umit Cali, Abdolhossein Sarrafzadeh, Ali Karimoddini, Afef Fekih

**Abstract**: This paper proposes a cyber-resilient secure control framework for autonomous vehicles (AVs) subject to false data injection (FDI) threats as actuator attacks. The framework integrates data-driven modeling, event-triggered communication, and fractional-order sliding mode control (FSMC) to enhance the resilience against adversarial interventions. A dynamic model decomposition (DMD)-based methodology is employed to extract the lateral dynamics from real-world data, eliminating the reliance on conventional mechanistic modeling. To optimize communication efficiency, an event-triggered transmission scheme is designed to reduce the redundant transmissions while ensuring system stability. Furthermore, an extended state observer (ESO) is developed for real-time estimation and mitigation of actuator attack effects. Theoretical stability analysis, conducted using Lyapunov methods and linear matrix inequality (LMI) formulations, guarantees exponential error convergence. Extensive simulations validate the proposed event-triggered secure control framework, demonstrating substantial improvements in attack mitigation, communication efficiency, and lateral tracking performance. The results show that the framework effectively counteracts actuator attacks while optimizing communication-resource utilization, making it highly suitable for safety-critical AV applications.

摘要: 本文针对自动驾驶汽车（AV）提出了一种具有网络弹性的安全控制框架，该框架受到致动器攻击的虚假数据注入（FDI）威胁。该框架集成了数据驱动建模、事件触发通信和分数阶滑动模式控制（FSMC），以增强对抗性干预的弹性。采用基于动态模型分解（DMZ）的方法从现实世界数据中提取横向动力学，消除了对传统机械建模的依赖。为了优化通信效率，设计了事件触发传输方案，以减少冗余传输，同时确保系统稳定性。此外，还开发了扩展状态观测器（ESO）来实时估计和缓解致动器攻击效应。使用李雅普诺夫方法和线性矩阵不等式（LDI）公式进行的理论稳定性分析保证了指数误差收敛。广泛的模拟验证了拟议的事件触发安全控制框架，展示了攻击缓解、通信效率和横向跟踪性能方面的重大改进。结果表明，该框架有效地对抗致动器攻击，同时优化通信资源利用率，使其非常适合安全关键的AV应用。



## **38. BreakFun: Jailbreaking LLMs via Schema Exploitation**

BreakFun：通过模式利用越狱LLM cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2510.17904v2) [paper-pdf](https://arxiv.org/pdf/2510.17904v2)

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.

摘要: 大型语言模型（LLM）在处理结构化数据和遵守语法规则方面的熟练程度是推动其广泛采用的一种能力，但也使它们变得脆弱。在本文中，我们通过BreakFun调查了这个漏洞，BreakFun是一种越狱方法，可以将LLM对结构化模式的遵守武器化。BreakFun采用了一个由三部分组成的提示，将一个无辜的框架和一个思想链分散注意力与一个核心“特洛伊模式”相结合-一个精心制作的数据结构，迫使模型生成有害内容，利用LLM遵循结构和模式的强烈倾向。我们证明该漏洞具有高度可转移性，JailbreakBench上的13个基础和专有模型平均成功率为89%，并在几个著名模型上达到100%的攻击成功率（ASB）。一项严格的消融研究证实，该特洛伊模式是攻击的主要原因。为了解决这个问题，我们引入了对抗性提示解构护栏，这是一种利用二级LLM来执行“文字转录”的防御--提取所有人类可读的文本以隔离和揭示用户真正的有害意图。我们的概念验证护栏展示了针对攻击的高功效，验证了针对欺骗性模式是一种可行的缓解策略。我们的工作探讨了法学硕士的核心优势如何转化为关键弱点，为构建更稳健一致的模型提供了新的视角。



## **39. DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation with Uncertainty-Quantification Support**

DGTON：一个鲁棒的基于深高斯的图神经网络，用于动态信任评估，并支持不确定性量化 cs.LG

15 pages, 6 figures, 5 tables

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2510.07620v2) [paper-pdf](https://arxiv.org/pdf/2510.07620v2)

**Authors**: Muhammad Usman, Yugyung Lee

**Abstract**: Dynamic trust evaluation in large, rapidly evolving graphs demands models that capture changing relationships, express calibrated confidence, and resist adversarial manipulation. DGTEN (Deep Gaussian-Based Trust Evaluation Network) introduces a unified graph-based framework that does all three by combining uncertainty-aware message passing, expressive temporal modeling, and built-in defenses against trust-targeted attacks. It represents nodes and edges as Gaussian distributions so that both semantic signals and epistemic uncertainty propagate through the graph neural network, enabling risk-aware trust decisions rather than overconfident guesses. To track how trust evolves, it layers hybrid absolute-Gaussian-hourglass positional encoding with Kolmogorov-Arnold network-based unbiased multi-head attention, then applies an ordinary differential equation-based residual learning module to jointly model abrupt shifts and smooth trends. Robust adaptive ensemble coefficient analysis prunes or down-weights suspicious interactions using complementary cosine and Jaccard similarity, curbing reputation laundering, sabotage, and on-off attacks. On two signed Bitcoin trust networks, DGTEN delivers standout gains where it matters most: in single-timeslot prediction on Bitcoin-OTC, it improves MCC by +12.34% over the best dynamic baseline; in the cold-start scenario on Bitcoin-Alpha, it achieves a +25.00% MCC improvement, the largest across all tasks and datasets; while under adversarial on-off attacks, it surpasses the baseline by up to +10.23% MCC. These results endorse the unified DGTEN framework.

摘要: 在大型、快速发展的图中进行动态信任评估，需要模型能够捕捉不断变化的关系，表达校准的置信度，并抵抗敌对操纵。DGTEN（Deep Gaussian-Based Trust Evaluation Network，基于深度高斯的信任评估网络）引入了一个统一的基于图的框架，通过结合不确定性感知的消息传递、表达性时间建模和针对信任目标攻击的内置防御来实现这三个目标。它将节点和边表示为高斯分布，以便语义信号和认知不确定性都通过图神经网络传播，从而实现风险感知的信任决策，而不是过度自信的猜测。为了跟踪信任如何演变，它将混合绝对高斯沙漏位置编码与基于Kolmogorov-Arnold网络的无偏多头注意力分层，然后应用基于常微分方程的残差学习模块来联合建模突变和平滑趋势。稳健的自适应集成系数分析使用互补cos和Jaccard相似性来修剪或淡化可疑的交互，从而遏制声誉洗钱、破坏和开关攻击。在两个已签署的比特币信任网络上，DGTON在最重要的地方取得了突出的收益：在Bitcoin-OTC的单时段预测中，它将MCC比最佳动态基线提高了+12.34%;在Bitcoin-Alpha的冷启动场景中，它实现了+25.00% MCC改进，这是所有任务和数据集中最大的;而在对抗性开关攻击下，它超出基线高达+10.23% MCC。这些结果认可了统一的DGTON框架。



## **40. Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification**

隐秘但有效：对图分类的保留分布后门攻击 cs.LG

Accepted by NeurIPS 2025

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2509.26032v2) [paper-pdf](https://arxiv.org/pdf/2509.26032v2)

**Authors**: Xiaobao Wang, Ruoxiao Sun, Yujun Zhang, Bingdao Feng, Dongxiao He, Luzhi Wang, Di Jin

**Abstract**: Graph Neural Networks (GNNs) have demonstrated strong performance across tasks such as node classification, link prediction, and graph classification, but remain vulnerable to backdoor attacks that implant imperceptible triggers during training to control predictions. While node-level attacks exploit local message passing, graph-level attacks face the harder challenge of manipulating global representations while maintaining stealth. We identify two main sources of anomaly in existing graph classification backdoor methods: structural deviation from rare subgraph triggers and semantic deviation caused by label flipping, both of which make poisoned graphs easily detectable by anomaly detection models. To address this, we propose DPSBA, a clean-label backdoor framework that learns in-distribution triggers via adversarial training guided by anomaly-aware discriminators. DPSBA effectively suppresses both structural and semantic anomalies, achieving high attack success while significantly improving stealth. Extensive experiments on real-world datasets validate that DPSBA achieves a superior balance between effectiveness and detectability compared to state-of-the-art baselines.

摘要: 图神经网络（GNN）在节点分类、链接预测和图分类等任务中表现出了强大的性能，但仍然容易受到后门攻击，这些后门攻击在训练过程中植入了不可感知的触发器来控制预测。虽然节点级攻击利用本地消息传递，但图级攻击面临着在保持隐身的同时操纵全局表示的更大挑战。我们确定了现有的图分类后门方法中的两个主要异常来源：罕见的子图触发器的结构偏差和标签翻转引起的语义偏差，这两个异常检测模型都可以很容易地检测到中毒图。为了解决这个问题，我们提出了DPSBA，这是一个干净标签的后门框架，它通过异常感知鉴别器指导的对抗训练来学习分发触发器。DPSBA有效抑制结构和语义异常，实现高攻击成功，同时显着提高隐身性。对现实世界数据集的广泛实验证实，与最先进的基线相比，DPBA在有效性和可检测性之间实现了更好的平衡。



## **41. LLMs Encode Harmfulness and Refusal Separately**

LLM分别编码伤害和拒绝 cs.CL

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2507.11878v4) [paper-pdf](https://arxiv.org/pdf/2507.11878v4)

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.

摘要: LLM接受过拒绝有害指令的培训，但他们真正了解除了拒绝之外的危害吗？先前的工作表明，LLM的拒绝行为可以通过一维子空间来调节，即拒绝方向。在这项工作中，我们确定了一个新的维度来分析LLM中的安全机制，即危害性，它在内部被编码为与拒绝分开的概念。存在一个与拒绝方向不同的危害方向。作为因果证据，沿着有害方向引导可能会导致LLM将无害的指令解释为有害的，但沿着拒绝方向引导往往会直接引发拒绝反应，而不会扭转模型对有害性的判断。此外，使用我们确定的危害性概念，我们发现某些越狱方法通过减少拒绝信号来发挥作用，而不会扭转模型内部的危害性信念。我们还发现，对模型进行不利调整以接受有害指令对模型内部有害性信念的影响最小。这些见解导致了一个实际的安全应用：该模型的潜在危害表示可以作为一个内在的保障（潜在的警卫），用于检测不安全的输入和减少过度拒绝，这是强大的微调攻击。例如，我们的潜伏卫士在不同的越狱方法中实现了与Llama Guard 3 8 B相当或更好的性能，Llama Guard 3 8 B是一种专用的微调保护模型。我们的研究结果表明，LLM对危害性的内部理解比他们拒绝不同输入指令的决定更强大，为研究AI安全性提供了一个新的视角。



## **42. DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective**

Databench：从对抗角度评估深度学习中的数据集审计 cs.CR

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2507.05622v3) [paper-pdf](https://arxiv.org/pdf/2507.05622v3)

**Authors**: Shuo Shao, Yiming Li, Mengren Zheng, Zhiyang Hu, Yukun Chen, Boheng Li, Yu He, Junfeng Guo, Dacheng Tao, Zhan Qin

**Abstract**: The widespread application of Deep Learning across diverse domains hinges critically on the quality and composition of training datasets. However, the common lack of disclosure regarding their usage raises significant privacy and copyright concerns. Dataset auditing techniques, which aim to determine if a specific dataset was used to train a given suspicious model, provide promising solutions to addressing these transparency gaps. While prior work has developed various auditing methods, their resilience against dedicated adversarial attacks remains largely unexplored. To bridge the gap, this paper initiates a comprehensive study evaluating dataset auditing from an adversarial perspective. We start with introducing a novel taxonomy, classifying existing methods based on their reliance on internal features (IF) (inherent to the data) versus external features (EF) (artificially introduced for auditing). Subsequently, we formulate two primary attack types: evasion attacks, designed to conceal the use of a dataset, and forgery attacks, intending to falsely implicate an unused dataset. Building on the understanding of existing methods and attack objectives, we further propose systematic attack strategies: decoupling, removal, and detection for evasion; adversarial example-based methods for forgery. These formulations and strategies lead to our new benchmark, DATABench, comprising 17 evasion attacks, 5 forgery attacks, and 9 representative auditing methods. Extensive evaluations using DATABench reveal that none of the evaluated auditing methods are sufficiently robust or distinctive under adversarial settings. These findings underscore the urgent need for developing a more secure and reliable dataset auditing method capable of withstanding sophisticated adversarial manipulation. Code is available in https://github.com/shaoshuo-ss/DATABench.

摘要: 深度学习在不同领域的广泛应用关键取决于训练数据集的质量和组成。然而，普遍缺乏对其使用情况的披露，引发了严重的隐私和版权问题。数据集审计技术旨在确定特定数据集是否用于训练给定的可疑模型，为解决这些透明度差距提供了有希望的解决方案。虽然之前的工作已经开发了各种审计方法，但它们对专门对抗攻击的弹性在很大程度上仍未被探索。为了弥合这一差距，本文发起了一项全面的研究，从对抗的角度评估数据集审计。我们首先引入一种新颖的分类法，根据现有方法对内部特征（IF）（数据固有的）和外部特征（EF）（人为引入以进行审计）的依赖来对现有方法进行分类。随后，我们制定了两种主要的攻击类型：逃避攻击（旨在隐藏数据集的使用）和伪造攻击（旨在错误地暗示未使用的数据集）。在对现有方法和攻击目标的理解的基础上，我们进一步提出了系统性攻击策略：脱钩、删除和检测逃避;基于对抗性示例的伪造方法。这些公式和策略导致了我们的新基准Databench，其中包括17种规避攻击、5种伪造攻击和9种代表性审计方法。使用Databench进行的广泛评估表明，在对抗环境下，所评估的审计方法都不够稳健或独特。这些发现凸显了开发一种能够承受复杂对抗操纵的更安全、更可靠的数据集审计方法的迫切需要。代码可在https://github.com/shaoshuo-ss/DATABench上获取。



## **43. Meta Policy Switching for Secure UAV Deconfliction in Adversarial Airspace**

对抗空域安全无人机冲突的Meta策略切换 cs.LG

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2506.21127v3) [paper-pdf](https://arxiv.org/pdf/2506.21127v3)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: Autonomous UAV navigation using reinforcement learning (RL) is vulnerable to adversarial attacks that manipulate sensor inputs, potentially leading to unsafe behavior and mission failure. Although robust RL methods provide partial protection, they often struggle to generalize to unseen or out-of-distribution (OOD) attacks due to their reliance on fixed perturbation settings. To address this limitation, we propose a meta-policy switching framework in which a meta-level polic dynamically selects among multiple robust policies to counter unknown adversarial shifts. At the core of this framework lies a discounted Thompson sampling (DTS) mechanism that formulates policy selection as a multi-armed bandit problem, thereby minimizing value distribution shifts via self-induced adversarial observations. We first construct a diverse ensemble of action-robust policies trained under varying perturbation intensities. The DTS-based meta-policy then adaptively selects among these policies online, optimizing resilience against self-induced, piecewise-stationary attacks. Theoretical analysis shows that the DTS mechanism minimizes expected regret, ensuring adaptive robustness to OOD attacks and exhibiting emergent antifragile behavior under uncertainty. Extensive simulations in complex 3D obstacle environments under both white-box (Projected Gradient Descent) and black-box (GPS spoofing) attacks demonstrate significantly improved navigation efficiency and higher conflict free trajectory rates compared to standard robust and vanilla RL baselines, highlighting the practical security and dependability benefits of the proposed approach.

摘要: 使用强化学习（RL）的自主无人机导航很容易受到操纵传感器输入的对抗攻击，可能导致不安全行为和任务失败。尽管稳健的RL方法提供了部分保护，但由于它们依赖于固定的扰动设置，它们通常很难推广到不可见或非分布（OOD）攻击。为了解决这一限制，我们提出了一个元政策切换框架，其中元级别政策在多个稳健的政策中动态选择，以应对未知的对抗性转变。该框架的核心是折扣汤普森抽样（NPS）机制，该机制将政策选择制定为多臂强盗问题，从而通过自引发的对抗性观察最大限度地减少价值分布变化。我们首先构建了在不同扰动强度下训练的行动稳健政策的多样化集合。然后，基于DTS的元策略在线自适应地选择这些策略，优化针对自引发的、分段稳定攻击的弹性。理论分析表明，RST机制最大限度地减少了预期遗憾，确保了对OOD攻击的自适应鲁棒性，并在不确定性下表现出紧急反脆弱行为。在白盒（投影梯度下降）和黑匣子（GPS欺骗）攻击下，在复杂的3D障碍环境中进行了广泛的模拟，证明了与标准稳健和普通RL基线相比，导航效率显着提高和无冲突轨迹率更高，凸显了所提出方法的实际安全性和可靠性优势。



## **44. A Flat Minima Perspective on Understanding Augmentations and Model Robustness**

理解增强和模型稳健性的平坦极小值观点 cs.LG

In Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI) 2026, Singapore

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2505.24592v3) [paper-pdf](https://arxiv.org/pdf/2505.24592v3)

**Authors**: Weebum Yoo, Sung Whan Yoon

**Abstract**: Model robustness indicates a model's capability to generalize well on unforeseen distributional shifts, including data corruptions and adversarial attacks. Data augmentation is one of the most prevalent and effective ways to enhance robustness. Despite the great success of the diverse augmentations in different fields, a unified theoretical understanding of their efficacy in improving model robustness is lacking. We theoretically reveal a general condition for label-preserving augmentations to bring robustness to diverse distribution shifts through the lens of flat minima and generalization bound, which de facto turns out to be strongly correlated with robustness against different distribution shifts in practice. Unlike most earlier works, our theoretical framework accommodates all the label-preserving augmentations and is not limited to particular distribution shifts. We substantiate our theories through different simulations on the existing common corruption and adversarial robustness benchmarks based on the CIFAR and ImageNet datasets.

摘要: 模型稳健性表明模型能够很好地概括不可预见的分布变化，包括数据损坏和对抗性攻击。数据增强是增强稳健性最普遍、最有效的方法之一。尽管不同领域的不同增强措施取得了巨大成功，但对其提高模型稳健性的功效缺乏统一的理论理解。我们从理论上揭示了标签保留增强的一般条件，通过平坦极小值和概括界限的视角，为多样化分布转变带来鲁棒性，事实上，这与实践中针对不同分布转变的鲁棒性密切相关。与大多数早期作品不同，我们的理论框架容纳了所有保留标签的增强，并且不限于特定的分布转变。我们基于CIFAR和ImageNet数据集，通过对现有常见腐败和对抗稳健性基准的不同模拟来证实我们的理论。



## **45. Quantum Support Vector Regression for Robust Anomaly Detection**

量子支持量回归用于鲁棒异常检测 quant-ph

Accepted to International Conference on Agents and Artificial Intelligence (ICAART) 2026

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2505.01012v3) [paper-pdf](https://arxiv.org/pdf/2505.01012v3)

**Authors**: Kilian Tscharke, Maximilian Wendlinger, Sebastian Issel, Pascal Debus

**Abstract**: Anomaly Detection (AD) is critical in data analysis, particularly within the domain of IT security. In this study, we explore the potential of Quantum Machine Learning for application to AD with special focus on the robustness to noise and adversarial attacks. We build upon previous work on Quantum Support Vector Regression (QSVR) for semisupervised AD by conducting a comprehensive benchmark on IBM quantum hardware using eleven datasets. Our results demonstrate that QSVR achieves strong classification performance and even outperforms the noiseless simulation on two of these datasets. Moreover, we investigate the influence of - in the NISQ-era inevitable - quantum noise on the performance of the QSVR. Our findings reveal that the model exhibits robustness to depolarizing, phase damping, phase flip, and bit flip noise, while amplitude damping and miscalibration noise prove to be more disruptive. Finally, we explore the domain of Quantum Adversarial Machine Learning by demonstrating that QSVR is highly vulnerable to adversarial attacks, with neither quantum noise nor adversarial training improving the model's robustness against such attacks.

摘要: 异常检测（AD）在数据分析中至关重要，尤其是在IT安全领域。在这项研究中，我们探索了量子机器学习应用于AD的潜力，特别关注对噪音和对抗攻击的鲁棒性。我们在之前针对半监督AD的量子支持量回归（QSVR）工作的基础上，使用11个数据集对IBM量子硬件进行了全面的基准测试。我们的结果表明，QSVR实现了强大的分类性能，甚至优于对其中两个数据集的无噪模拟。此外，我们还研究了在NISQ时代不可避免的量子噪音对QSVR性能的影响。我们的研究结果表明，该模型对去极化、相衰减、相翻转和位翻转噪音表现出鲁棒性，而幅度衰减和失调噪音被证明更具破坏性。最后，我们通过证明QSVR极易受到对抗攻击来探索量子对抗机器学习领域，量子噪音和对抗训练都没有提高模型对此类攻击的鲁棒性。



## **46. Towards Backdoor Stealthiness in Model Parameter Space**

模型参数空间中的后门隐秘性 cs.CR

accepted by CCS 2025

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2501.05928v3) [paper-pdf](https://arxiv.org/pdf/2501.05928v3)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Stjepan Picek

**Abstract**: Recent research on backdoor stealthiness focuses mainly on indistinguishable triggers in input space and inseparable backdoor representations in feature space, aiming to circumvent backdoor defenses that examine these respective spaces. However, existing backdoor attacks are typically designed to resist a specific type of backdoor defense without considering the diverse range of defense mechanisms. Based on this observation, we pose a natural question: Are current backdoor attacks truly a real-world threat when facing diverse practical defenses?   To answer this question, we examine 12 common backdoor attacks that focus on input-space or feature-space stealthiness and 17 diverse representative defenses. Surprisingly, we reveal a critical blind spot: Backdoor attacks designed to be stealthy in input and feature spaces can be mitigated by examining backdoored models in parameter space. To investigate the underlying causes behind this common vulnerability, we study the characteristics of backdoor attacks in the parameter space. Notably, we find that input- and feature-space attacks introduce prominent backdoor-related neurons in parameter space, which are not thoroughly considered by current backdoor attacks. Taking comprehensive stealthiness into account, we propose a novel supply-chain attack called Grond. Grond limits the parameter changes by a simple yet effective module, Adversarial Backdoor Injection (ABI), which adaptively increases the parameter-space stealthiness during the backdoor injection. Extensive experiments demonstrate that Grond outperforms all 12 backdoor attacks against state-of-the-art (including adaptive) defenses on CIFAR-10, GTSRB, and a subset of ImageNet. In addition, we show that ABI consistently improves the effectiveness of common backdoor attacks.

摘要: 最近关于后门隐蔽性的研究主要关注输入空间中不可区分的触发器和特征空间中不可分割的后门表示，旨在规避检查这些各自空间的后门防御。然而，现有的后门攻击通常旨在抵抗特定类型的后门防御，而不考虑各种防御机制。基于这一观察，我们提出了一个自然的问题：当面临多样化的实际防御时，当前的后门攻击真的是现实世界的威胁吗？   为了回答这个问题，我们研究了12种常见的后门攻击，这些攻击重点关注输入空间或特征空间的隐蔽性，以及17种不同的代表性防御。令人惊讶的是，我们揭示了一个关键的盲点：设计在输入和特征空间中隐蔽的后门攻击可以通过检查参数空间中的后门模型来缓解。为了调查这种常见漏洞背后的根本原因，我们研究了参数空间中后门攻击的特征。值得注意的是，我们发现输入和特征空间攻击在参数空间中引入了突出的后门相关神经元，而当前的后门攻击并没有彻底考虑这些神经元。考虑到全面的隐秘性，我们提出了一种名为Grond的新型供应链攻击。Grond通过一个简单而有效的模块对抗后门注入（ABI）限制参数变化，该模块自适应地增加后门注入期间的参数空间隐蔽性。大量实验表明，针对CIFAR-10、GTSRB和ImageNet子集的最先进（包括自适应）防御，Grond的性能优于所有12种后门攻击。此外，我们还表明，ABI始终提高了常见后门攻击的有效性。



## **47. FlippedRAG: Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models**

FlippedRAG：黑盒意见操纵对抗性攻击检索增强生成模型 cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2501.02968v5) [paper-pdf](https://arxiv.org/pdf/2501.02968v5)

**Authors**: Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) enriches LLMs by dynamically retrieving external knowledge, reducing hallucinations and satisfying real-time information needs. While existing research mainly targets RAG's performance and efficiency, emerging studies highlight critical security concerns. Yet, current adversarial approaches remain limited, mostly addressing white-box scenarios or heuristic black-box attacks without fully investigating vulnerabilities in the retrieval phase. Additionally, prior works mainly focus on factoid Q&A tasks, their attacks lack complexity and can be easily corrected by advanced LLMs. In this paper, we investigate a more realistic and critical threat scenario: adversarial attacks intended for opinion manipulation against black-box RAG models, particularly on controversial topics. Specifically, we propose FlippedRAG, a transfer-based adversarial attack against black-box RAG systems. We first demonstrate that the underlying retriever of a black-box RAG system can be reverse-engineered, enabling us to train a surrogate retriever. Leveraging the surrogate retriever, we further craft target poisoning triggers, altering vary few documents to effectively manipulate both retrieval and subsequent generation. Extensive empirical results show that FlippedRAG substantially outperforms baseline methods, improving the average attack success rate by 16.7%. FlippedRAG achieves on average a 50% directional shift in the opinion polarity of RAG-generated responses, ultimately causing a notable 20% shift in user cognition. Furthermore, we evaluate the performance of several potential defensive measures, concluding that existing mitigation strategies remain insufficient against such sophisticated manipulation attacks. These results highlight an urgent need for developing innovative defensive solutions to ensure the security and trustworthiness of RAG systems.

摘要: 检索增强生成（RAG）通过动态检索外部知识、减少幻觉和满足实时信息需求来丰富LLM。虽然现有研究主要针对RAG的性能和效率，但新出现的研究强调了关键的安全问题。然而，当前的对抗方法仍然有限，主要解决白盒场景或启发式黑匣子攻击，而没有在检索阶段充分调查漏洞。此外，之前的作品主要集中在事实问答任务上，其攻击缺乏复杂性，并且可以通过高级LLM轻松纠正。在本文中，我们研究了一个更现实和关键的威胁场景：针对黑箱RAG模型的意见操纵的对抗性攻击，特别是在有争议的话题上。具体来说，我们提出了FlippedRAG，这是一种针对黑盒RAG系统的基于传输的对抗性攻击。我们首先证明了一个黑盒RAG系统的底层检索器可以进行逆向工程，使我们能够训练一个代理检索器。利用代理检索器，我们进一步工艺目标中毒触发器，改变不同的几个文件，以有效地操纵检索和后续生成。广泛的实证结果表明，FlippedRAG的性能大大优于基线方法，将平均攻击成功率提高了16.7%。FlippedRAG平均实现了RAG生成的响应的意见两极50%的方向性转变，最终导致用户认知发生了20%的显着转变。此外，我们评估了几种潜在防御措施的性能，得出的结论是，现有的缓解策略仍然不足以应对此类复杂的操纵攻击。这些结果凸显了开发创新防御解决方案的迫切需要，以确保RAG系统的安全性和可信性。



## **48. Defending Collaborative Filtering Recommenders via Adversarial Robustness Based Edge Reweighting**

基于对抗鲁棒性的边缘重权协同过滤推荐防御 cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2412.10850v2) [paper-pdf](https://arxiv.org/pdf/2412.10850v2)

**Authors**: Yongyu Wang

**Abstract**: User based collaborative filtering (CF) relies on a user and user similarity graph, making it vulnerable to profile injection (shilling) attacks that manipulate neighborhood relations to promote (push) or demote (nuke) target items. In this work, we propose an adversarial robustness based edge reweighting defense for CF. We first assign each user and user edge a non robustness score via spectral adversarial robustness evaluation, which quantifies the edge sensitivity to adversarial perturbations. We then attenuate the influence of non robust edges by reweighting similarities during prediction. Extensive experiments demonstrate that the proposed method effectively defends against various types of attacks.

摘要: 基于用户的协作过滤（CF）依赖于用户和用户相似性图，因此容易受到配置文件注入（先令）攻击，这些攻击操纵邻居关系以升级（推送）或降级（核武器）目标项目。在这项工作中，我们提出了一种基于对抗鲁棒性的CF边缘重加权防御。我们首先通过谱对抗稳健性评估为每个用户和用户边分配非稳健性分数，该评估量化了边缘对对抗扰动的敏感性。然后，我们通过在预测期间重新加权相似性来削弱非稳健边缘的影响。大量实验表明，该方法可以有效防御各种类型的攻击。



## **49. Certifying Robustness of Graph Convolutional Networks for Node Perturbation with Polyhedra Abstract Interpretation**

用多边形抽象解释证明图卷积网络对节点扰动的鲁棒性 cs.LG

Author preprint, published at Data Mining and Knowledge Discovery. Published version available at: https://link.springer.com/article/10.1007/s10618-025-01180-w

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2405.08645v2) [paper-pdf](https://arxiv.org/pdf/2405.08645v2)

**Authors**: Boqi Chen, Kristóf Marussy, Oszkár Semeráth, Gunter Mussbacher, Dániel Varró

**Abstract**: Graph convolutional neural networks (GCNs) are powerful tools for learning graph-based knowledge representations from training data. However, they are vulnerable to small perturbations in the input graph, which makes them susceptible to input faults or adversarial attacks. This poses a significant problem for GCNs intended to be used in critical applications, which need to provide certifiably robust services even in the presence of adversarial perturbations. We propose an improved GCN robustness certification technique for node classification in the presence of node feature perturbations. We introduce a novel polyhedra-based abstract interpretation approach to tackle specific challenges of graph data and provide tight upper and lower bounds for the robustness of the GCN. Experiments show that our approach simultaneously improves the tightness of robustness bounds as well as the runtime performance of certification. Moreover, our method can be used during training to further improve the robustness of GCNs.

摘要: 图卷积神经网络（GCN）是从训练数据学习基于图的知识表示的强大工具。然而，它们很容易受到输入图中的微小扰动的影响，这使得它们容易受到输入错误或对抗攻击。这对于旨在用于关键应用的GCN构成了一个重大问题，即使存在对抗性扰动，这些应用也需要提供可认证的稳健服务。我们提出了一种改进的GCN鲁棒性认证技术，用于存在节点特征扰动的节点分类。我们引入了一种新的基于多面体的抽象解释方法来解决图数据的特定挑战，并为GCN的鲁棒性提供严格的上限和下限。实验表明，我们的方法同时提高了鲁棒性边界的紧密性以及认证的运行时性能。此外，我们的方法可以在训练过程中使用，以进一步提高GCN的鲁棒性。



## **50. We Can Always Catch You: Detecting Adversarial Patched Objects WITH or WITHOUT Signature**

我们总是可以抓住你：检测带有或不带有签名的敌对修补对象 cs.CV

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2106.05261v3) [paper-pdf](https://arxiv.org/pdf/2106.05261v3)

**Authors**: Jiachun Li, Jianan Feng, Jianjun Huang, Bin Liang

**Abstract**: Recently, object detection has proven vulnerable to adversarial patch attacks. The attackers holding a specially crafted patch can hide themselves from state-of-the-art detectors, e.g., YOLO, even in the physical world. This attack can bring serious security threats, such as escaping from surveillance cameras. How to effectively detect this kind of adversarial examples to catch potential attacks has become an important problem. In this paper, we propose two detection methods: the signature-based method and the signature-independent method. First, we identify two signatures of existing adversarial patches that can be utilized to precisely locate patches within adversarial examples. By employing the signatures, a fast signature-based method is developed to detect the adversarial objects. Second, we present a robust signature-independent method based on the \textit{content semantics consistency} of model outputs. Adversarial objects violate this consistency, appearing locally but disappearing globally, while benign ones remain consistently present. The experiments demonstrate that two proposed methods can effectively detect attacks both in the digital and physical world. These methods each offer distinct advantage. Specifically, the signature-based method is capable of real-time detection, while the signature-independent method can detect unknown adversarial patch attacks and makes defense-aware attacks almost impossible to perform.

摘要: 最近，对象检测已被证明容易受到对抗补丁攻击。持有特制补丁的攻击者可以隐藏自己，免受最先进的检测器的侵害，例如，YOLO，即使在物理世界中也是如此。这种攻击可能会带来严重的安全威胁，例如逃离监控摄像头。如何有效检测此类对抗性示例以捕捉潜在的攻击已成为一个重要问题。本文提出了两种检测方法：基于签名的方法和独立签名的方法。首先，我们识别现有对抗性补丁的两个签名，可用于在对抗性示例中准确定位补丁。通过利用这些签名，开发了一种基于签名的快速方法来检测对抗对象。其次，我们提出了一种基于模型输出的\textit{内容语义一致性}的稳健签名无关方法。敌对对象违反了这种一致性，出现在局部但在全球范围内消失，而良性对象则始终存在。实验表明，提出的两种方法可以有效检测数字和物理世界中的攻击。这些方法都具有独特的优势。具体来说，基于签名的方法能够实时检测，而独立签名的方法可以检测未知的对抗性补丁攻击，并使防御感知攻击几乎不可能执行。



