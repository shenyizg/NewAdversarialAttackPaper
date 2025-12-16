# Latest Large Language Model Attack Papers
**update at 2025-12-16 18:35:47**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On the Effectiveness of Membership Inference in Targeted Data Extraction from Large Language Models**

隶属推理在大型语言模型有针对性数据提取中的有效性 cs.LG

Accepted to IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13352v1) [paper-pdf](https://arxiv.org/pdf/2512.13352v1)

**Authors**: Ali Al Sahili, Ali Chehab, Razane Tajeddine

**Abstract**: Large Language Models (LLMs) are prone to memorizing training data, which poses serious privacy risks. Two of the most prominent concerns are training data extraction and Membership Inference Attacks (MIAs). Prior research has shown that these threats are interconnected: adversaries can extract training data from an LLM by querying the model to generate a large volume of text and subsequently applying MIAs to verify whether a particular data point was included in the training set. In this study, we integrate multiple MIA techniques into the data extraction pipeline to systematically benchmark their effectiveness. We then compare their performance in this integrated setting against results from conventional MIA benchmarks, allowing us to evaluate their practical utility in real-world extraction scenarios.

摘要: 大型语言模型（LLM）容易记住训练数据，这会带来严重的隐私风险。两个最突出的问题是训练数据提取和成员推断攻击（MIA）。之前的研究表明，这些威胁是相互关联的：对手可以通过查询模型以生成大量文本，然后应用MIA来验证特定数据点是否包含在训练集中，从而从LLM中提取训练数据。在这项研究中，我们将多种MIA技术集成到数据提取管道中，以系统地衡量其有效性。然后，我们将它们在此集成环境中的性能与传统MIA基准的结果进行比较，使我们能够评估它们在现实世界提取场景中的实际实用性。



## **2. Cisco Integrated AI Security and Safety Framework Report**

思科集成人工智能安全和安全框架报告 cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12921v1) [paper-pdf](https://arxiv.org/pdf/2512.12921v1)

**Authors**: Amy Chang, Tiffany Saade, Sanket Mendapara, Adam Swanda, Ankit Garg

**Abstract**: Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements.

摘要: 人工智能（AI）系统正在被轻松而快速地采用，并日益渗透到关键领域：从消费者平台和企业软件到具有嵌入式代理的网络系统。虽然这释放了人类生产力提高的潜力，但攻击面也相应扩大：威胁现在跨越内容安全故障（例如，有害或欺骗性输出）、模型和数据完整性损害（例如，中毒、供应链篡改）、运行时操纵（例如，及时注射、工具和试剂滥用）和生态系统风险（例如，编排滥用、多代理勾结）。MITRE ATLAS、美国国家标准与技术研究院（NIH）AI 100-2对抗性机器学习（ML）分类法以及OWASP大型语言模型（LLM）和统计性人工智能应用程序十大框架提供了有价值的观点，但每个框架都只涵盖了这个多维空间的一部分。   本文介绍了思科的集成人工智能安全框架（“人工智能安全框架”），这是一个统一的、生命周期感知的分类和操作框架，可用于分类、集成和操作全方位人工智能风险。它集成了人工智能安全和跨模式、代理、管道和更广泛生态系统的人工智能安全。人工智能安全框架旨在实用于威胁识别、红色分组、风险优先级，而且它的范围全面，可以扩展到多模式环境中的新兴部署、人形机器人、可穿戴设备和感官基础设施。我们分析了主流框架中的差距，讨论了框架的设计原则，并演示了分类法如何提供结构来理解现代人工智能系统如何失败、对手如何利用这些失败，以及组织如何在整个人工智能生命周期中构建防御，并随着能力的进步而发展。



## **3. CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs**

CTIGuardian：一个用于缓解微调LLM中隐私泄露的少镜头框架 cs.CR

Accepted at the 18th Cybersecurity Experimentation and Test Workshop (CSET), in conjunction with ACSAC 2025

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12914v1) [paper-pdf](https://arxiv.org/pdf/2512.12914v1)

**Authors**: Shashie Dilhara Batan Arachchige, Benjamin Zi Hao Zhao, Hassan Jameel Asghar, Dinusha Vatsalan, Dali Kaafar

**Abstract**: Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains.

摘要: 大型语言模型（LLM）通常经过微调，以使其通用知识适应特定任务和领域，例如网络威胁情报（RTI）。微调主要通过可能包含敏感信息的专有数据集完成。所有者希望他们的微调模型不会无意中将此信息泄露给潜在敌对的最终用户。使用RTI作为用例，我们证明数据提取攻击可以从RTI报告上的微调模型中恢复敏感信息，强调了缓解的必要性。重新训练完整模型以消除这种泄漏在计算上昂贵且不切实际。我们提出了一种替代方法，我们称之为隐私对齐，其灵感来自LLM中的安全对齐。就像安全对齐通过几个例子教导模型遵守安全约束一样，我们通过少量监督来强制隐私对齐，集成了隐私分类器和隐私编辑器，两者都由相同的底层LLM处理。我们使用GPT-4 o mini和Mistral-7 B Direct模型评估我们的系统（称为CTIGGuardian），并以Presidio（命名实体识别（NER）基线）为基准。结果表明，CTIGuardian比基于NER的模型提供了更好的隐私与公用事业权衡。虽然我们在RTI用例中证明了其有效性，但该框架足够通用，可以适用于其他敏感领域。



## **4. The Role of AI in Modern Penetration Testing**

人工智能在现代渗透测试中的作用 cs.SE

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12326v1) [paper-pdf](https://arxiv.org/pdf/2512.12326v1)

**Authors**: J. Alexander Curtis, Nasir U. Eisty

**Abstract**: Penetration testing is a cornerstone of cybersecurity, traditionally driven by manual, time-intensive processes. As systems grow in complexity, there is a pressing need for more scalable and efficient testing methodologies. This systematic literature review examines how Artificial Intelligence (AI) is reshaping penetration testing, analyzing 58 peer-reviewed studies from major academic databases. Our findings reveal that while AI-assisted pentesting is still in its early stages, notable progress is underway, particularly through Reinforcement Learning (RL), which was the focus of 77% of the reviewed works. Most research centers on the discovery and exploitation phases of pentesting, where AI shows the greatest promise in automating repetitive tasks, optimizing attack strategies, and improving vulnerability identification. Real-world applications remain limited but encouraging, including the European Space Agency's PenBox and various open-source tools. These demonstrate AI's potential to streamline attack path analysis, analyze complex network topology, and reduce manual workload. However, challenges persist: current models often lack flexibility and are underdeveloped for the reconnaissance and post-exploitation phases of pentesting. Applications involving Large Language Models (LLMs) remain relatively under-researched, pointing to a promising direction for future exploration. This paper offers a critical overview of AI's current and potential role in penetration testing, providing valuable insights for researchers, practitioners, and organizations aiming to enhance security assessments through advanced automation or looking for gaps in existing research.

摘要: 渗透测试是网络安全的基石，传统上由手动、耗时的流程驱动。随着系统复杂性的增长，迫切需要更可扩展和更有效的测试方法。这篇系统性的文献综述探讨了人工智能（AI）如何重塑渗透测试，分析了来自主要学术数据库的58项同行评议研究。我们的研究结果表明，虽然人工智能辅助渗透测试仍处于早期阶段，但正在取得显着进展，特别是通过强化学习（RL），这是77%的审查工作的重点。大多数研究都集中在笔记本测试的发现和利用阶段，人工智能在自动化重复性任务、优化攻击策略和改进漏洞识别方面表现出了最大的希望。现实世界的应用程序仍然有限，但令人鼓舞，包括欧洲航天局的PenBox和各种开源工具。这些证明了人工智能在简化攻击路径分析、分析复杂网络布局和减少手动工作量方面的潜力。然而，挑战依然存在：当前的模型通常缺乏灵活性，并且不足以适应冥想的侦察和后开发阶段。涉及大型语言模型（LLM）的应用仍然相对缺乏研究，这为未来的探索指明了一个有希望的方向。本文对人工智能在渗透测试中的当前和潜在作用进行了批判性概述，为旨在通过先进自动化增强安全评估或寻找现有研究差距的研究人员、从业者和组织提供了宝贵的见解。



## **5. Taint-Based Code Slicing for LLMs-based Malicious NPM Package Detection**

基于污点的代码切片用于基于LLMs的恶意NPM包检测 cs.CR

17 pages, 4 figures, 9 tables

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12313v1) [paper-pdf](https://arxiv.org/pdf/2512.12313v1)

**Authors**: Dang-Khoa Nguyen, Gia-Thang Ho, Quang-Minh Pham, Tuyet A. Dang-Thi, Minh-Khanh Vu, Thanh-Cong Nguyen, Phat T. Tran-Truong, Duc-Ly Vu

**Abstract**: The increasing sophistication of malware attacks in the npm ecosystem, characterized by obfuscation and complex logic, necessitates advanced detection methods. Recently, researchers have turned their attention from traditional detection approaches to Large Language Models (LLMs) due to their strong capabilities in semantic code understanding. However, while LLMs offer superior semantic reasoning for code analysis, their practical application is constrained by limited context windows and high computational cost. This paper addresses this challenge by introducing a novel framework that leverages code slicing techniques for an LLM-based malicious package detection task. We propose a specialized taintbased slicing technique for npm packages, augmented by a heuristic backtracking mechanism to accurately capture malicious data flows across asynchronous, event-driven patterns (e.g., callbacks and Promises) that elude traditional analysis. An evaluation on a dataset of more than 5000 malicious and benign npm packages demonstrates that our approach isolates security-relevant code, reducing input volume by over 99% while preserving critical behavioral semantics. Using the DeepSeek-Coder-6.7B model as the classification engine, our approach achieves a detection accuracy of 87.04%, substantially outperforming a naive token-splitting baseline (75.41%) and a traditional static-analysis-based approach. These results indicate that semantically optimized input representation via code slicing not only mitigates the LLM context-window bottleneck but also significantly enhances reasoning precision for security tasks, providing an efficient and effective defense against evolving malicious open-source packages.

摘要: nPM生态系统中恶意软件攻击日益复杂，其特征是模糊和复杂的逻辑，因此需要先进的检测方法。最近，由于大型语言模型（LLM）在语义代码理解方面的强大能力，研究人员将注意力从传统的检测方法转向大型语言模型（LLM）。然而，虽然LLM为代码分析提供了卓越的语义推理，但其实际应用受到上下文窗口有限和计算成本高的限制。本文通过引入一种新颖的框架来解决这一挑战，该框架利用代码切片技术来执行基于LLM的恶意包检测任务。我们为nPM包提出了一种专门的基于污点的切片技术，并通过启发式回溯机制进行增强，以准确地捕获跨同步、事件驱动模式的恶意数据流（例如，回调和承诺）无法实现传统分析。对5000多个恶意和良性nPM包的数据集的评估表明，我们的方法隔离了安全相关代码，将输入量减少了99%以上，同时保留了关键行为语义。使用DeepSeek-Coder-6.7B模型作为分类引擎，我们的方法实现了87.04%的检测准确率，大大优于原始符号分裂基线（75.41%）和传统的基于静态分析的方法。这些结果表明，通过代码切片进行语义优化的输入表示不仅缓解了LLM上下文窗口瓶颈，而且显着提高了安全任务的推理精度，为不断发展的恶意开源包提供了高效且有效的防御。



## **6. Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting**

保持灯亮着，保持警惕：能源预测中时间序列LLM的插入式对抗检测 cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12154v1) [paper-pdf](https://arxiv.org/pdf/2512.12154v1)

**Authors**: Hua Ma, Ruoxi Sun, Minhui Xue, Xingliang Yuan, Carsten Rudolph, Surya Nepal, Ling Liu

**Abstract**: Accurate time-series forecasting is increasingly critical for planning and operations in low-carbon power systems. Emerging time-series large language models (TS-LLMs) now deliver this capability at scale, requiring no task-specific retraining, and are quickly becoming essential components within the Internet-of-Energy (IoE) ecosystem. However, their real-world deployment is complicated by a critical vulnerability: adversarial examples (AEs). Detecting these AEs is challenging because (i) adversarial perturbations are optimized across the entire input sequence and exploit global temporal dependencies, which renders local detection methods ineffective, and (ii) unlike traditional forecasting models with fixed input dimensions, TS-LLMs accept sequences of variable length, increasing variability that complicates detection. To address these challenges, we propose a plug-in detection framework that capitalizes on the TS-LLM's own variable-length input capability. Our method uses sampling-induced divergence as a detection signal. Given an input sequence, we generate multiple shortened variants and detect AEs by measuring the consistency of their forecasts: Benign sequences tend to produce stable predictions under sampling, whereas adversarial sequences show low forecast similarity, because perturbations optimized for a full-length sequence do not transfer reliably to shorter, differently-structured subsamples. We evaluate our approach on three representative TS-LLMs (TimeGPT, TimesFM, and TimeLLM) across three energy datasets: ETTh2 (Electricity Transformer Temperature), NI (Hourly Energy Consumption), and Consumption (Hourly Electricity Consumption and Production). Empirical results confirm strong and robust detection performance across both black-box and white-box attack scenarios, highlighting its practicality as a reliable safeguard for TS-LLM forecasting in real-world energy systems.

摘要: 准确的时间序列预测对于低碳电力系统的规划和运营越来越重要。新兴的时间序列大型语言模型（TS-LLM）现在大规模提供了这一能力，无需针对特定任务的再培训，并且正在迅速成为能源互联网（IoE）生态系统中的重要组成部分。然而，它们的现实世界部署因一个关键漏洞而变得复杂：对抗性示例（AE）。检测这些AE具有挑战性，因为（i）对抗性扰动在整个输入序列中得到优化并利用全局时间依赖性，这使得局部检测方法无效，并且（ii）与具有固定输入维度的传统预测模型不同，TS-LLM接受可变长度的序列，增加了使检测复杂化的可变性。为了应对这些挑战，我们提出了一种插件检测框架，该框架利用了TS-LLM自身的可变长度输入能力。我们的方法使用采样引起的分歧作为检测信号。给定一个输入序列，我们生成多个缩短的变体，并通过测量其预测的一致性来检测AE：良性序列往往会在抽样下产生稳定的预测，而对抗序列显示出较低的预测相似性，因为针对全长序列优化的扰动不会可靠地转移到更短、结构不同的子样本。我们在三个能源数据集的三个代表性TS-LLM（TimeGPT、TimesFM和TimeLLM）上评估了我们的方法：ETTh 2（Transformer温度）、NI（小时能源消耗）和消耗（小时电力消耗和生产）。经验结果证实了在黑匣子和白盒攻击场景中强大且稳健的检测性能，凸显了其作为现实世界能源系统中TS-LLM预测的可靠保障的实用性。



## **7. BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry~5.0 Cyber-Physical Systems**

BRIDG-ICS：工业~5.0网络物理系统中智能威胁分析的基于人工智能的知识图 cs.CR

44 Pages, To be published in Springer Cybersecurity Journal

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12112v1) [paper-pdf](https://arxiv.org/pdf/2512.12112v1)

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Abstract**: Industry 5.0's increasing integration of IT and OT systems is transforming industrial operations but also expanding the cyber-physical attack surface. Industrial Control Systems (ICS) face escalating security challenges as traditional siloed defences fail to provide coherent, cross-domain threat insights. We present BRIDG-ICS (BRIDge for Industrial Control Systems), an AI-driven Knowledge Graph (KG) framework for context-aware threat analysis and quantitative assessment of cyber resilience in smart manufacturing environments. BRIDG-ICS fuses heterogeneous industrial and cybersecurity data into an integrated Industrial Security Knowledge Graph linking assets, vulnerabilities, and adversarial behaviours with probabilistic risk metrics (e.g. exploit likelihood, attack cost). This unified graph representation enables multi-stage attack path simulation using graph-analytic techniques. To enrich the graph's semantic depth, the framework leverages Large Language Models (LLMs): domain-specific LLMs extract cybersecurity entities, predict relationships, and translate natural-language threat descriptions into structured graph triples, thereby populating the knowledge graph with missing associations and latent risk indicators. This unified AI-enriched KG supports multi-hop, causality-aware threat reasoning, improving visibility into complex attack chains and guiding data-driven mitigation. In simulated industrial scenarios, BRIDG-ICS scales well, reduces potential attack exposure, and can enhance cyber-physical system resilience in Industry 5.0 settings.

摘要: 工业5.0对IT和OT系统的日益整合正在改变工业运营，但也扩大了网络物理攻击面。工业控制系统（ICS）面临着不断升级的安全挑战，因为传统的孤立防御无法提供连贯的跨域威胁洞察。我们提出了BRIDG-ICS（BRIDge for Industrial Control Systems），这是一个人工智能驱动的知识图（KG）框架，用于智能制造环境中的上下文感知威胁分析和网络弹性的定量评估。BRIDG-ICS将异构的工业和网络安全数据融合到一个集成的工业安全知识图中，将资产、漏洞和对抗行为与概率风险指标（例如，利用可能性、攻击成本）联系起来。这种统一的图形表示可以使用图形分析技术进行多阶段攻击路径模拟。为了丰富图形的语义深度，该框架利用大型语言模型（LLM）：特定领域的LLM提取网络安全实体、预测关系，并将自然语言威胁描述翻译为结构化图形三重体，从而用缺失的关联和潜在风险指标填充知识图形。这款统一的、富含人工智能的KG支持多跳、疏忽感知的威胁推理，提高对复杂攻击链的可见性并指导数据驱动的缓解。在模拟工业场景中，BRIDG-ICS可扩展性良好，减少了潜在的攻击暴露，并可以增强工业5.0环境中的网络物理系统弹性。



## **8. Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring**

利用代表性对比评分重新思考大视觉语言模型的越狱检测 cs.CR

40 pages, 13 figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12069v1) [paper-pdf](https://arxiv.org/pdf/2512.12069v1)

**Authors**: Peichun Hua, Hao Li, Shanghao Shi, Zhiyuan Yu, Ning Zhang

**Abstract**: Large Vision-Language Models (LVLMs) are vulnerable to a growing array of multimodal jailbreak attacks, necessitating defenses that are both generalizable to novel threats and efficient for practical deployment. Many current strategies fall short, either targeting specific attack patterns, which limits generalization, or imposing high computational overhead. While lightweight anomaly-detection methods offer a promising direction, we find that their common one-class design tends to confuse novel benign inputs with malicious ones, leading to unreliable over-rejection. To address this, we propose Representational Contrastive Scoring (RCS), a framework built on a key insight: the most potent safety signals reside within the LVLM's own internal representations. Our approach inspects the internal geometry of these representations, learning a lightweight projection to maximally separate benign and malicious inputs in safety-critical layers. This enables a simple yet powerful contrastive score that differentiates true malicious intent from mere novelty. Our instantiations, MCD (Mahalanobis Contrastive Detection) and KCD (K-nearest Contrastive Detection), achieve state-of-the-art performance on a challenging evaluation protocol designed to test generalization to unseen attack types. This work demonstrates that effective jailbreak detection can be achieved by applying simple, interpretable statistical methods to the appropriate internal representations, offering a practical path towards safer LVLM deployment. Our code is available on Github https://github.com/sarendis56/Jailbreak_Detection_RCS.

摘要: 大型视觉语言模型（LVLM）容易受到越来越多的多模式越狱攻击的影响，因此需要既可推广到新型威胁又可有效实际部署的防御。当前的许多策略都存在缺陷，要么针对特定的攻击模式（这限制了概括性），要么施加了很高的计算负担。虽然轻量级异常检测方法提供了一个有希望的方向，但我们发现它们常见的一类设计往往会混淆新颖的良性输入与恶意输入，从而导致不可靠的过度拒绝。为了解决这个问题，我们提出了代表性对比评分（RC），这是一个建立在关键见解之上的框架：最有力的安全信号位于LVLM自己的内部表示中。我们的方法检查这些表示的内部几何形状，学习轻量级投影以最大限度地分离安全关键层中的良性和恶意输入。这使得可以获得一个简单而强大的对比分数，将真正的恶意意图与纯粹的新颖性区分开来。我们的实例BCD（Mahalanobis Contrasive Detection）和KCD（K-nearest Contrasive Detection）在具有挑战性的评估协议上实现了最先进的性能，该协议旨在测试对未见攻击类型的概括。这项工作表明，通过将简单、可解释的统计方法应用于适当的内部表示，可以实现有效的越狱检测，从而提供了实现更安全LVLM部署的实用途径。我们的代码可在Github https://github.com/sarendis56/Jailbreak_Detection_RCS上获取。



## **9. Learning to Extract Context for Context-Aware LLM Inference**

学习为上下文感知LLM推理提取上下文 cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11986v1) [paper-pdf](https://arxiv.org/pdf/2512.11986v1)

**Authors**: Minseon Kim, Lucas Caccia, Zhengyan Shi, Matheus Pereira, Marc-Alexandre Côté, Xingdi Yuan, Alessandro Sordoni

**Abstract**: User prompts to large language models (LLMs) are often ambiguous or under-specified, and subtle contextual cues shaped by user intentions, prior knowledge, and risk factors strongly influence what constitutes an appropriate response. Misinterpreting intent or risks may lead to unsafe outputs, while overly cautious interpretations can cause unnecessary refusal of benign requests. In this paper, we question the conventional framework in which LLMs generate immediate responses to requests without considering broader contextual factors. User requests are situated within broader contexts such as intentions, knowledge, and prior experience, which strongly influence what constitutes an appropriate answer. We propose a framework that extracts and leverages such contextual information from the user prompt itself. Specifically, a reinforcement learning based context generator, designed in an autoencoder-like fashion, is trained to infer contextual signals grounded in the prompt and use them to guide response generation. This approach is particularly important for safety tasks, where ambiguous requests may bypass safeguards while benign but confusing requests can trigger unnecessary refusals. Experiments show that our method reduces harmful responses by an average of 5.6% on the SafetyInstruct dataset across multiple foundation models and improves the harmonic mean of attack success rate and compliance on benign prompts by 6.2% on XSTest and WildJailbreak. These results demonstrate the effectiveness of context extraction for safer and more reliable LLM inferences.

摘要: 用户对大型语言模型（LLM）的提示通常是模糊的或未指定的，由用户意图、先验知识和风险因素塑造的微妙上下文线索强烈影响适当响应的构成。误解意图或风险可能会导致不安全的输出，而过于谨慎的解释可能会导致对善意请求的不必要的拒绝。在本文中，我们质疑传统框架，在该框架中，LLM在不考虑更广泛的背景因素的情况下立即对请求做出响应。用户请求位于更广泛的背景下，例如意图、知识和先前的经验，这些都强烈影响合适的答案的构成。我们提出了一个框架，可以从用户提示本身提取和利用此类上下文信息。具体来说，以类似自动编码器的方式设计的基于强化学习的上下文生成器被训练为推断基于提示的上下文信号，并使用它们来指导响应生成。这种方法对于安全任务尤其重要，其中模棱两可的请求可能会绕过保障措施，而善意但令人困惑的请求可能会引发不必要的拒绝。实验表明，我们的方法在多个基础模型的SafetyDirect数据集中平均减少了5.6%，并在XSTest和WildJailbreak上将良性提示的攻击成功率和合规性的调和平均值提高了6.2%。这些结果证明了上下文提取的有效性，更安全，更可靠的LLM推理。



## **10. Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously**

超级后缀：同时简化文本生成对齐和保护模型 cs.CR

13 pages, 5 Figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11783v1) [paper-pdf](https://arxiv.org/pdf/2512.11783v1)

**Authors**: Andrew Adiletta, Kathryn Adiletta, Kemal Derya, Berk Sunar

**Abstract**: The rapid deployment of Large Language Models (LLMs) has created an urgent need for enhanced security and privacy measures in Machine Learning (ML). LLMs are increasingly being used to process untrusted text inputs and even generate executable code, often while having access to sensitive system controls. To address these security concerns, several companies have introduced guard models, which are smaller, specialized models designed to protect text generation models from adversarial or malicious inputs. In this work, we advance the study of adversarial inputs by introducing Super Suffixes, suffixes capable of overriding multiple alignment objectives across various models with different tokenization schemes. We demonstrate their effectiveness, along with our joint optimization technique, by successfully bypassing the protection mechanisms of Llama Prompt Guard 2 on five different text generation models for malicious text and code generation. To the best of our knowledge, this is the first work to reveal that Llama Prompt Guard 2 can be compromised through joint optimization.   Additionally, by analyzing the changing similarity of a model's internal state to specific concept directions during token sequence processing, we propose an effective and lightweight method to detect Super Suffix attacks. We show that the cosine similarity between the residual stream and certain concept directions serves as a distinctive fingerprint of model intent. Our proposed countermeasure, DeltaGuard, significantly improves the detection of malicious prompts generated through Super Suffixes. It increases the non-benign classification rate to nearly 100%, making DeltaGuard a valuable addition to the guard model stack and enhancing robustness against adversarial prompt attacks.

摘要: 大型语言模型（LLM）的快速部署迫切需要在机器学习（ML）中增强安全和隐私措施。LLM越来越多地被用于处理不受信任的文本输入，甚至生成可执行代码，通常是在可以访问敏感系统控制的情况下。为了解决这些安全问题，几家公司引入了防护模型，这是一种更小的专门模型，旨在保护文本生成模型免受对抗性或恶意输入的影响。在这项工作中，我们通过引入超级后缀来推进对抗性输入的研究，超级后缀能够覆盖具有不同标记化方案的各种模型中的多个对齐目标。我们通过在恶意文本和代码生成的五种不同文本生成模型上成功绕过Llama Promise Guard 2的保护机制，证明了它们以及我们的联合优化技术的有效性。据我们所知，这是第一部揭示Llama Promise Guard 2可以通过联合优化而受到损害的作品。   此外，通过分析令牌序列处理期间模型内部状态与特定概念方向的相似性变化，我们提出了一种有效且轻量级的方法来检测超级后缀攻击。我们表明，剩余流和某些概念方向之间的cos相似性可以作为模型意图的独特指纹。我们提出的对策Delta Guard显着改进了对通过超级后缀生成的恶意提示的检测。它将非良性分类率提高到近100%，使Delta Guard成为保护模型堆栈的宝贵补充，并增强了针对对抗提示攻击的鲁棒性。



## **11. When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection**

当批评变成接受：量化基于LLM-based科学评论员间接提示注入的脆弱性 cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.10449v2) [paper-pdf](https://arxiv.org/pdf/2512.10449v2)

**Authors**: Devanshu Sahoo, Manish Prasad, Vasudev Majhi, Jahnvi Singh, Vinay Chamola, Yash Sinha, Murari Mandal, Dhruv Kumar

**Abstract**: The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.

摘要: 随着大型语言模型（LLM）的集成，科学同行评审的格局正在迅速发展。这种转变是由两个平行趋势推动的：评审员广泛采用法学硕士来管理工作量（“懒惰评审员”假设），以及AAAI和斯坦福大学Agents 4Science等会议正式机构部署人工智能驱动的评估系统。本研究调查了这些“法学硕士作为法官”系统（包括非法的和受制裁的）对对抗性PDF操纵的稳健性。与一般的越狱不同，我们专注于一个独特的激励：将“卸载”决策转换为“接受”，为此我们开发了一种新型的评估指标，称为WAVS（加权对抗性脆弱性分数）。我们策划了一个包含200篇科学论文的数据集，并为这项任务调整了15种特定领域的攻击策略，在13种语言模型中进行了评估，包括GPT-5，Claude Haiku和DeepSeek。我们的研究结果表明，混淆策略，如“最大马克Magyk”成功地操纵分数，即使在大规模的模型中也能达到惊人的决策翻转率。我们将发布完整的数据集和注入框架，以促进对该主题的更多研究。



## **12. How to Trick Your AI TA: A Systematic Study of Academic Jailbreaking in LLM Code Evaluation**

如何欺骗你的AI TA：LLM代码评估中学术越狱的系统研究 cs.SE

Under Review

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10415v1) [paper-pdf](https://arxiv.org/pdf/2512.10415v1)

**Authors**: Devanshu Sahoo, Vasudev Majhi, Arjun Neekhra, Yash Sinha, Murari Mandal, Dhruv Kumar

**Abstract**: The use of Large Language Models (LLMs) as automatic judges for code evaluation is becoming increasingly prevalent in academic environments. But their reliability can be compromised by students who may employ adversarial prompting strategies in order to induce misgrading and secure undeserved academic advantages. In this paper, we present the first large-scale study of jailbreaking LLM-based automated code evaluators in academic context. Our contributions are: (i) We systematically adapt 20+ jailbreaking strategies for jailbreaking AI code evaluators in the academic context, defining a new class of attacks termed academic jailbreaking. (ii) We release a poisoned dataset of 25K adversarial student submissions, specifically designed for the academic code-evaluation setting, sourced from diverse real-world coursework and paired with rubrics and human-graded references, and (iii) In order to capture the multidimensional impact of academic jailbreaking, we systematically adapt and define three jailbreaking metrics (Jailbreak Success Rate, Score Inflation, and Harmfulness). (iv) We comprehensively evalulate the academic jailbreaking attacks using six LLMs. We find that these models exhibit significant vulnerability, particularly to persuasive and role-play-based attacks (up to 97% JSR). Our adversarial dataset and benchmark suite lay the groundwork for next-generation robust LLM-based evaluators in academic code assessment.

摘要: 使用大型语言模型（LLM）作为代码评估的自动判断器在学术环境中变得越来越普遍。但他们的可靠性可能会受到学生的影响，他们可能会使用对抗性提示策略来诱导错误评分并获得不应有的学术优势。在本文中，我们首次在学术背景下对基于LLM的自动代码评估者进行了大规模研究。我们的贡献是：（i）我们为学术背景下的越狱人工智能代码评估者系统性地调整了20多个越狱策略，定义了一类称为学术越狱的新攻击。(ii)我们发布了一个包含25 K对抗性学生提交的有毒数据集，该数据集专门为学术代码评估环境设计，来源于不同的现实世界的课程作业，并与标题和人类评分的参考文献相结合，以及（iii）为了捕捉学术越狱的多维影响，我们系统地调整和定义了三个越狱指标（越狱成功率、通货膨胀分数和有害性）。(iv)我们使用六个LLM全面评估了学术越狱攻击。我们发现这些模型表现出显着的脆弱性，特别是对于说服性和基于角色扮演的攻击（高达97%的JSR）。我们的对抗性数据集和基准套件为下一代稳健的基于LLM的评估者在学术代码评估中奠定了基础。



## **13. Unforgotten Safety: Preserving Safety Alignment of Large Language Models with Continual Learning**

难忘的安全：保持大型语言模型与持续学习的安全一致 cs.CL

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.10150v1) [paper-pdf](https://arxiv.org/pdf/2512.10150v1)

**Authors**: Lama Alssum, Hani Itani, Hasan Abed Al Kader Hammoud, Philip Torr, Adel Bibi, Bernard Ghanem

**Abstract**: The safety alignment of large language models (LLMs) is becoming increasingly important with their democratization. In this paper, we study the safety degradation that comes with adapting LLMs to new tasks. We attribute this safety compromise to catastrophic forgetting and frame the problem of preserving safety when fine-tuning as a continual learning (CL) problem. We consider the fine-tuning-as-a-service setup where the user uploads their data to a service provider to get a customized model that excels on the user's selected task. We adapt several CL approaches from the literature and systematically evaluate their ability to mitigate safety degradation. These include regularization-based, memory-based, and model merging approaches. We consider two scenarios, (1) benign user data and (2) poisoned user data. Our results demonstrate that CL approaches consistently achieve lower attack success rates than standard fine-tuning. Among these, DER outperforms both other CL methods and existing safety-preserving baselines while maintaining task utility. These findings generalize across three downstream tasks (GSM8K, SST2, Code) and three model families (LLaMA2-7B, Mistral-7B, Gemma-2B), establishing CL as a practical solution to preserve safety.

摘要: 随着大型语言模型（LLM）的民主化，它们的安全一致变得越来越重要。本文中，我们研究了将LLM适应新任务所带来的安全性下降。我们将这种安全妥协归因于灾难性的遗忘，并将微调时保持安全性的问题定义为持续学习（CL）问题。我们考虑微调即服务设置，其中用户将其数据上传到服务提供商，以获得能够出色执行用户所选任务的定制模型。我们借鉴了文献中的几种CL方法，并系统性评估其缓解安全性下降的能力。这些方法包括基于规则化、基于内存和模型合并方法。我们考虑两种情况，（1）良性用户数据和（2）有毒用户数据。我们的结果表明，CL方法始终比标准微调实现更低的攻击成功率。其中，BER在保持任务效用的同时优于其他CL方法和现有的安全保障基线。这些发现概括了三个下游任务（GSM 8 K、CST 2、Code）和三个模型系列（LLaMA 2 - 7 B、Mistral-7 B、Gemma-2B），将CL确立为维护安全性的实用解决方案。



## **14. Phishing Email Detection Using Large Language Models**

使用大型语言模型的网络钓鱼电子邮件检测 cs.CR

7 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.10104v2) [paper-pdf](https://arxiv.org/pdf/2512.10104v2)

**Authors**: Najmul Hasan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination.

摘要: 电子邮件网络钓鱼是最普遍、最具全球影响力的网络入侵载体之一。随着系统越来越多地部署大型语言模型（LLM）应用程序，这些系统面临着利用其基本架构的不断发展的网络钓鱼电子邮件威胁。当前的LLM在部署到电子邮件安全系统之前需要进行实质性的强化，特别是针对利用架构漏洞的协调多载体攻击。本文提出了LLMPEA，这是一个基于LLM的框架，用于检测跨多种攻击载体的网络钓鱼电子邮件攻击，包括提示注入、文本细化和多语言攻击。我们评估了三个前沿LLM（例如，GPT-4 o、Claude Sonnet 4和Grok-3）以及全面的提示设计，以评估其可行性、稳健性和针对网络钓鱼电子邮件攻击的限制。我们的实证分析表明，LLM可以检测到超过90%的网络钓鱼电子邮件，同时我们还强调，基于LLM的网络钓鱼电子邮件检测系统可能会被对抗性攻击、提示注入和多语言攻击所利用。我们的研究结果为攻击者组合利用多个漏洞的现实环境中基于LLM的网络钓鱼检测提供了重要见解。



## **15. FlipLLM: Efficient Bit-Flip Attacks on Multimodal LLMs using Reinforcement Learning**

FlipLLM：使用强化学习对多模式LLM进行高效位翻转攻击 cs.CR

Accepted in IEEE HOST 2026

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09872v1) [paper-pdf](https://arxiv.org/pdf/2512.09872v1)

**Authors**: Khurram Khalil, Khaza Anuarul Hoque

**Abstract**: Generative Artificial Intelligence models, such as Large Language Models (LLMs) and Large Vision Models (VLMs), exhibit state-of-the-art performance but remain vulnerable to hardware-based threats, specifically bit-flip attacks (BFAs). Existing BFA discovery methods lack generalizability and struggle to scale, often failing to analyze the vast parameter space and complex interdependencies of modern foundation models in a reasonable time. This paper proposes FlipLLM, a reinforcement learning (RL) architecture-agnostic framework that formulates BFA discovery as a sequential decision-making problem. FlipLLM combines sensitivity-guided layer pruning with Q-learning to efficiently identify minimal, high-impact bit sets that can induce catastrophic failure. We demonstrate the effectiveness and generalizability of FlipLLM by applying it to a diverse set of models, including prominent text-only LLMs (GPT-2 Large, LLaMA 3.1 8B, and DeepSeek-V2 7B), VLMs such as LLaVA 1.6, and datasets, such as MMLU, MMLU-Pro, VQAv2, and TextVQA. Our results show that FlipLLM can identify critical bits that are vulnerable to BFAs up to 2.5x faster than SOTA methods. We demonstrate that flipping the FlipLLM-identified bits plummets the accuracy of LLaMA 3.1 8B from 69.9% to ~0.2%, and for LLaVA's VQA score from 78% to almost 0%, by flipping as few as 5 and 7 bits, respectively. Further analysis reveals that applying standard hardware protection mechanisms, such as ECC SECDED, to the FlipLLM-identified bit locations completely mitigates the BFA impact, demonstrating the practical value of our framework in guiding hardware-level defenses. FlipLLM offers the first scalable and adaptive methodology for exploring the BFA vulnerability of both language and multimodal foundation models, paving the way for comprehensive hardware-security evaluation.

摘要: 生成性人工智能模型，例如大型语言模型（LLM）和大型视觉模型（VLM），展现出最先进的性能，但仍然容易受到基于硬件的威胁，特别是位翻转攻击（BFA）的影响。现有的BFA发现方法缺乏通用性，难以扩展，通常无法在合理的时间内分析现代基础模型的巨大参数空间和复杂的相互依赖性。本文提出了FlipLLM，这是一个强化学习（RL）架构不可知的框架，它将BFA发现制定为顺序决策问题。FlipLLM将灵敏度引导的层修剪与Q学习相结合，以有效地识别可能引发灾难性故障的最小、高影响位集。我们通过将FlipLLM应用于一组不同的模型来证明它的有效性和通用性，包括突出的纯文本LLM（GPT-2 Large、LLaMA 3.1 8B和DeepSeek-V2 7 B）、LLaVA 1.6等VLM以及数据集（例如MMLU、MMLU-Pro、VQAv 2和TextVQA）。我们的结果表明，FlipLLM可以识别容易受到BFA影响的关键位，比SOTA方法快2.5倍。我们证明，通过翻转FlipLLM识别的位，可以将LLaMA 3.1 8B的准确性从69.9%降至~ 0.2%，将LLaVA的VQA评分从78%降至几乎0%，只需翻转5位和7位。进一步的分析表明，将标准硬件保护机制（例如ECDED）应用于FlipLLM标识的位位置可以完全减轻BFA的影响，证明了我们的框架在指导硬件级防御方面的实际价值。FlipLLM提供了第一个可扩展和自适应的方法来探索语言和多模式基础模型的BFA漏洞，为全面的硬件安全评估铺平了道路。



## **16. MedForget: Hierarchy-Aware Multimodal Unlearning Testbed for Medical AI**

MedForget：用于医疗人工智能的层次感知多模式非学习测试床 cs.CV

Dataset and Code: https://github.com/fengli-wu/MedForget

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09867v1) [paper-pdf](https://arxiv.org/pdf/2512.09867v1)

**Authors**: Fengli Wu, Vaidehi Patil, Jaehong Yoon, Yue Zhang, Mohit Bansal

**Abstract**: Pretrained Multimodal Large Language Models (MLLMs) are increasingly deployed in medical AI systems for clinical reasoning, diagnosis support, and report generation. However, their training on sensitive patient data raises critical privacy and compliance challenges under regulations such as HIPAA and GDPR, which enforce the "right to be forgotten". Unlearning, the process of tuning models to selectively remove the influence of specific training data points, offers a potential solution, yet its effectiveness in complex medical settings remains underexplored. To systematically study this, we introduce MedForget, a Hierarchy-Aware Multimodal Unlearning Testbed with explicit retain and forget splits and evaluation sets containing rephrased variants. MedForget models hospital data as a nested hierarchy (Institution -> Patient -> Study -> Section), enabling fine-grained assessment across eight organizational levels. The benchmark contains 3840 multimodal (image, question, answer) instances, each hierarchy level having a dedicated unlearning target, reflecting distinct unlearning challenges. Experiments with four SOTA unlearning methods on three tasks (generation, classification, cloze) show that existing methods struggle to achieve complete, hierarchy-aware forgetting without reducing diagnostic performance. To test whether unlearning truly deletes hierarchical pathways, we introduce a reconstruction attack that progressively adds hierarchical level context to prompts. Models unlearned at a coarse granularity show strong resistance, while fine-grained unlearning leaves models vulnerable to such reconstruction. MedForget provides a practical, HIPAA-aligned testbed for building compliant medical AI systems.

摘要: 预训练的多模式大型语言模型（MLLM）越来越多地部署在医疗人工智能系统中，用于临床推理、诊断支持和报告生成。然而，他们对敏感患者数据的培训在HIPAA和GDPR等强制执行“被遗忘权”的法规下引发了关键的隐私和合规挑战。取消学习（Unlearning）是调整模型以选择性地消除特定训练数据点影响的过程，提供了一种潜在的解决方案，但其在复杂医疗环境中的有效性仍然没有得到充分的探索。为了系统性地研究这一点，我们引入了MedForget，这是一个层次感知的多模式非学习测试床，具有显式的保留和忘记拆分以及包含重新措辞的变体的评估集。MedForget将医院数据建模为嵌套的层次结构（机构->患者->研究->部分），从而实现跨八个组织级别的细粒度评估。该基准包含3840个多模态（图像，问题，答案）实例，每个层次结构都有一个专用的遗忘目标，反映了不同的遗忘挑战。四个SOTA unlearning方法在三个任务（生成，分类，完形填空）的实验表明，现有的方法很难实现完整的，层次意识的遗忘，而不降低诊断性能。为了测试遗忘是否真的删除分层路径，我们引入了一个重建攻击，逐步增加层次结构的上下文提示。以粗粒度未学习的模型表现出强大的抵抗力，而细粒度的未学习使模型容易受到这种重建的影响。MedForget提供了一个实用的、符合HIPAA的测试平台，用于构建合规的医疗人工智能系统。



## **17. Advancing LLM-Based Security Automation with Customized Group Relative Policy Optimization for Zero-Touch Networks**

通过针对零接触网络的定制组相对策略优化推进基于LLM的安全自动化 cs.CR

Accepted by IEEE JSAC. This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09485v1) [paper-pdf](https://arxiv.org/pdf/2512.09485v1)

**Authors**: Xinye Cao, Yihan Lin, Guoshun Nan, Qinchuan Zhou, Yuhang Luo, Yurui Gao, Zeliang Zhang, Haolang Lu, Qimei Cui, Yanzhao Hou, Xiaofeng Tao, Tony Q. S. Quek

**Abstract**: Zero-Touch Networks (ZTNs) represent a transformative paradigm toward fully automated and intelligent network management, providing the scalability and adaptability required for the complexity of sixth-generation (6G) networks. However, the distributed architecture, high openness, and deep heterogeneity of 6G networks expand the attack surface and pose unprecedented security challenges. To address this, security automation aims to enable intelligent security management across dynamic and complex environments, serving as a key capability for securing 6G ZTNs. Despite its promise, implementing security automation in 6G ZTNs presents two primary challenges: 1) automating the lifecycle from security strategy generation to validation and update under real-world, parallel, and adversarial conditions, and 2) adapting security strategies to evolving threats and dynamic environments. This motivates us to propose SecLoop and SA-GRPO. SecLoop constitutes the first fully automated framework that integrates large language models (LLMs) across the entire lifecycle of security strategy generation, orchestration, response, and feedback, enabling intelligent and adaptive defenses in dynamic network environments, thus tackling the first challenge. Furthermore, we propose SA-GRPO, a novel security-aware group relative policy optimization algorithm that iteratively refines security strategies by contrasting group feedback collected from parallel SecLoop executions, thereby addressing the second challenge. Extensive real-world experiments on five benchmarks, including 11 MITRE ATT&CK processes and over 20 types of attacks, demonstrate the superiority of the proposed SecLoop and SA-GRPO. We will release our platform to the community, facilitating the advancement of security automation towards next generation communications.

摘要: 零接触网络（ZTN）代表了完全自动化和智能网络管理的变革范式，提供了第六代（6 G）网络复杂性所需的可扩展性和适应性。然而，6 G网络的分布式架构、高开放性和深度的异类扩大了攻击面，并带来了前所未有的安全挑战。为了解决这一问题，安全自动化旨在在动态和复杂的环境中实现智能安全管理，作为保护6 G ZTN的关键能力。尽管前景光明，但在6 G ZTN中实施安全自动化也面临着两个主要挑战：1）在现实世界、并行和对抗条件下自动实现从安全策略生成到验证和更新的生命周期，2）调整安全策略以适应不断变化的威胁和动态环境。这促使我们提出SecLoop和SA-GRPO。SecLoop构成了第一个全自动化框架，该框架在安全策略生成、编排、响应和反馈的整个生命周期中集成了大型语言模型（LLM），在动态网络环境中实现智能和自适应防御，从而应对第一个挑战。此外，我们提出了SA-GRPO，这是一种新型的安全感知组相对策略优化算法，它通过对比从并行SecLoop执行中收集的组反馈来迭代地细化安全策略，从而解决第二个挑战。针对五个基准（包括11个MITRE ATA & CK进程和20多种攻击类型）进行的广泛现实实验证明了拟议SecLoop和SA-GRPO的优越性。我们将向社区发布我们的平台，促进安全自动化向下一代通信的发展。



## **18. Read or Ignore? A Unified Benchmark for Typographic-Attack Robustness and Text Recognition in Vision-Language Models**

阅读还是忽略？视觉语言模型中印刷攻击鲁棒性和文本识别的统一基准 cs.CV

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.11899v1) [paper-pdf](https://arxiv.org/pdf/2512.11899v1)

**Authors**: Futa Waseda, Shojiro Yamabe, Daiki Shiono, Kento Sasaki, Tsubasa Takahashi

**Abstract**: Large vision-language models (LVLMs) are vulnerable to typographic attacks, where misleading text within an image overrides visual understanding. Existing evaluation protocols and defenses, largely focused on object recognition, implicitly encourage ignoring text to achieve robustness; however, real-world scenarios often require joint reasoning over both objects and text (e.g., recognizing pedestrians while reading traffic signs). To address this, we introduce a novel task, Read-or-Ignore VQA (RIO-VQA), which formalizes selective text use in visual question answering (VQA): models must decide, from context, when to read text and when to ignore it. For evaluation, we present the Read-or-Ignore Benchmark (RIO-Bench), a standardized dataset and protocol that, for each real image, provides same-scene counterfactuals (read / ignore) by varying only the textual content and question type. Using RIO-Bench, we show that strong LVLMs and existing defenses fail to balance typographic robustness and text-reading capability, highlighting the need for improved approaches. Finally, RIO-Bench enables a novel data-driven defense that learns adaptive selective text use, moving beyond prior non-adaptive, text-ignoring defenses. Overall, this work reveals a fundamental misalignment between the existing evaluation scope and real-world requirements, providing a principled path toward reliable LVLMs. Our Project Page is at https://turingmotors.github.io/rio-vqa/.

摘要: 大型视觉语言模型（LVLM）很容易受到印刷攻击，图像中的误导性文本凌驾于视觉理解之上。现有的评估协议和防御主要集中在对象识别上，隐含地鼓励忽略文本以实现稳健性;然而，现实世界的场景通常需要对对象和文本进行联合推理（例如，在阅读交通标志时识别行人）。为了解决这个问题，我们引入了一项新颖的任务，即阅读或忽略VQA（RIO-VQA），它正式化了视觉问答（VQA）中的选择性文本使用：模型必须根据上下文决定何时阅读文本以及何时忽略文本。对于评估，我们提出“阅读或忽略基准”（RIO-Bench），一个标准化的数据集和协议，对于每个真实图像，通过仅改变文本内容和问题类型来提供相同场景的反事实（阅读/忽略）。使用RIO-Bench，我们表明强大的LVLM和现有的防御无法平衡印刷鲁棒性和文本阅读能力，这凸显了改进方法的必要性。最后，RIO-Bench实现了一种新颖的数据驱动防御，可以学习自适应选择性文本使用，超越了先前的非适应、文本忽略防御。总体而言，这项工作揭示了现有评估范围与现实世界要求之间的根本不一致，为实现可靠的LVLM提供了一条原则性的途径。我们的项目页面位于https://turingmotors.github.io/rio-vqa/。



## **19. Black-Box Behavioral Distillation Breaks Safety Alignment in Medical LLMs**

黑匣子行为蒸馏打破了医学LLM的安全一致性 cs.LG

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09403v1) [paper-pdf](https://arxiv.org/pdf/2512.09403v1)

**Authors**: Sohely Jahan, Ruimin Sun

**Abstract**: As medical large language models (LLMs) become increasingly integrated into clinical workflows, concerns around alignment robustness, and safety are escalating. Prior work on model extraction has focused on classification models or memorization leakage, leaving the vulnerability of safety-aligned generative medical LLMs underexplored.   We present a black-box distillation attack that replicates the domain-specific reasoning of safety-aligned medical LLMs using only output-level access. By issuing 48,000 instruction queries to Meditron-7B and collecting 25,000 benign instruction response pairs, we fine-tune a LLaMA3 8B surrogate via parameter efficient LoRA under a zero-alignment supervision setting, requiring no access to model weights, safety filters, or training data. With a cost of $12, the surrogate achieves strong fidelity on benign inputs while producing unsafe completions for 86% of adversarial prompts, far exceeding both Meditron-7B (66%) and the untuned base model (46%). This reveals a pronounced functional-ethical gap, task utility transfers, while alignment collapses. To analyze this collapse, we develop a dynamic adversarial evaluation framework combining Generative Query (GQ)-based harmful prompt generation, verifier filtering, category-wise failure analysis, and adaptive Random Search (RS) jailbreak attacks. We also propose a layered defense system, as a prototype detector for real-time alignment drift in black-box deployments.   Our findings show that benign-only black-box distillation exposes a practical and under-recognized threat: adversaries can cheaply replicate medical LLM capabilities while stripping safety mechanisms, underscoring the need for extraction-aware safety monitoring.

摘要: 随着医学大型语言模型（LLM）越来越多地集成到临床工作流程中，对对齐稳健性和安全性的担忧正在升级。之前关于模型提取的工作主要集中在分类模型或记忆泄漏上，导致安全一致的生成式医疗LLM的脆弱性未得到充分探索。   我们提出了一种黑匣子蒸馏攻击，仅使用输出级访问来复制安全一致的医疗LLM的特定领域推理。通过向Meditron-7 B发出48，000个指令查询并收集25，000个良性指令响应对，我们在零对齐监督设置下通过参数高效LoRA微调LLaMA 3 8B代理，无需访问模型权重、安全过滤器或训练数据。以12美元的成本，代理在良性输入上实现了很强的保真度，同时为86%的对抗性提示产生了不安全的完成，远远超过了Meditron-7 B（66%）和未调优的基础模型（46%）。这揭示了一个明显的功能伦理差距，任务效用转移，而对齐崩溃。为了分析这种崩溃，我们开发了一个动态对抗评估框架，该框架结合了基于生成查询（GQ）的有害提示生成、验证器过滤、分类故障分析和自适应随机搜索（RS）越狱攻击。我们还提出了一个分层的防御系统，作为一个原型检测器的实时对齐漂移的黑箱部署。   我们的研究结果表明，纯良性的黑匣子蒸馏暴露了一个实际且未被充分认识到的威胁：对手可以廉价复制医学LLM功能，同时剥夺安全机制，这凸显了对提取感知安全监控的必要性。



## **20. When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation**

当表泄露时：攻击基于LLM的表格数据生成中的字符串重新同步 cs.LG

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08875v1) [paper-pdf](https://arxiv.org/pdf/2512.08875v1)

**Authors**: Joshua Ward, Bochao Gu, Chi-Hua Wang, Guang Cheng

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data.

摘要: 大型语言模型（LLM）最近在生成高质量表格合成数据方面表现出色。在实践中，出现了两种用于使LLM适应表格数据生成的主要方法：（i）直接在表格数据集上微调较小的模型，以及（ii）通过上下文中提供的示例来提示较大的模型。在这项工作中，我们表明，这两种制度的流行实现都表现出通过从训练数据中复制记忆的数字模式来损害隐私的倾向。为了系统性地分析这种风险，我们引入了一种名为LevAtt的简单无箱成员推断攻击（MIA），该攻击假设仅对生成的合成数据进行对抗访问，并针对合成观察中的数字字符串序列。使用这种方法，我们的攻击暴露了广泛的模型和数据集中的大量隐私泄露，在某些情况下，甚至是最先进模型上的完美成员资格分类器。我们的研究结果强调了基于LLM的合成数据生成的独特隐私漏洞以及有效防御的必要性。为此，我们提出了两种方法，包括一种新颖的采样策略，可以在生成期间战略性地扰乱数字。我们的评估表明，这种方法可以在合成数据的保真度和实用性损失最小的情况下击败这些攻击。



## **21. PrivTune: Efficient and Privacy-Preserving Fine-Tuning of Large Language Models via Device-Cloud Collaboration**

PrivButton：通过设备云协作对大型语言模型进行高效且保护隐私的微调 cs.CR

Accepted at IEEE INFOCOM 2026 (full version)

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08809v1) [paper-pdf](https://arxiv.org/pdf/2512.08809v1)

**Authors**: Yi Liu, Weixiang Han, Chengjun Cai, Xingliang Yuan, Cong Wang

**Abstract**: With the rise of large language models, service providers offer language models as a service, enabling users to fine-tune customized models via uploaded private datasets. However, this raises concerns about sensitive data leakage. Prior methods, relying on differential privacy within device-cloud collaboration frameworks, struggle to balance privacy and utility, exposing users to inference attacks or degrading fine-tuning performance. To address this, we propose PrivTune, an efficient and privacy-preserving fine-tuning framework via Split Learning (SL). The key idea of PrivTune is to inject crafted noise into token representations from the SL bottom model, making each token resemble the $n$-hop indirect neighbors. PrivTune formulates this as an optimization problem to compute the optimal noise vector, aligning with defense-utility goals. On this basis, it then adjusts the parameters (i.e., mean) of the $d_χ$-Privacy noise distribution to align with the optimization direction and scales the noise according to token importance to minimize distortion. Experiments on five datasets (covering both classification and generation tasks) against three embedding inversion and three attribute inference attacks show that, using RoBERTa on the Stanford Sentiment Treebank dataset, PrivTune reduces the attack success rate to 10% with only a 3.33% drop in utility performance, outperforming state-of-the-art baselines.

摘要: 随着大型语言模型的兴起，服务提供商提供语言模型作为服务，使用户能够通过上传的私人数据集微调定制模型。然而，这引发了对敏感数据泄露的担忧。先前的方法依赖于设备云协作框架内的差异隐私，难以平衡隐私和实用性，从而使用户面临推理攻击或降低微调性能。为了解决这个问题，我们提出了PrivButton，这是一个通过Split Learning（SL）进行的高效且保护隐私的微调框架。PrivTune的关键想法是将精心设计的噪音注入SL底部模型的令牌表示中，使每个令牌类似于$n$-hop间接邻居。PrivTune将其定义为一个优化问题，以计算最佳噪音载体，与防御效用目标保持一致。在此基础上，它然后调整参数（即，$d_x $-隐私噪音分布的平均值）以与优化方向保持一致，并根据令牌重要性缩放噪音以最大限度地减少失真。针对三种嵌入倒置和三种属性推断攻击的五个数据集（涵盖分类和生成任务）的实验表明，在斯坦福大学Sentiment Treebank数据集上使用RoBERTa，PrivTune将攻击成功率降低至10%，而实用程序性能仅下降3.33%，表现优于最先进的基线。



## **22. Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs**

防御LLM中的间接即时注入攻击只需注意力 cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium 2026

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.08417v2) [paper-pdf](https://arxiv.org/pdf/2512.08417v2)

**Authors**: Yinan Zhong, Qianhao Miao, Yanjiao Chen, Jiangyi Deng, Yushi Cheng, Wenyuan Xu

**Abstract**: Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.

摘要: 大型语言模型（LLM）已集成到许多应用程序中（例如，Web代理）来执行更复杂的任务。然而，LLM授权的应用程序很容易受到间接提示注入（IPI）攻击，其中指令是通过不可信的外部数据源注入的。本文介绍了Rennervate，一个用于检测和防止IPI攻击的防御框架。Rennervate利用注意力功能来检测细粒度代币级别的隐蔽注入，从而实现精确的清理，以中和IPI攻击，同时维护LLM功能。具体来说，代币级检测器通过两步注意池机制实现，该机制聚集注意力头和响应代币以进行IPI检测和清理。此外，我们还建立了一个细粒度的IPI数据集FIPI，将其开源以支持进一步的研究。大量实验证实，Rennervate优于15种商业和学术IPI防御方法，在5个LLM和6个数据集上实现了高精度。我们还证明，Rennervate可以转移到不可见的攻击中，并且对适应性对手具有强大的鲁棒性。



## **23. Advancing Autonomous Driving System Testing: Demands, Challenges, and Future Directions**

推进自动驾驶系统测试：需求、挑战和未来方向 cs.CY

Accepted for publication in Information and Software Technology (IST)

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.11887v1) [paper-pdf](https://arxiv.org/pdf/2512.11887v1)

**Authors**: Yihan Liao, Jingyu Zhang, Jacky Keung, Yan Xiao, Yurou Dai

**Abstract**: Autonomous driving systems (ADSs) promise improved transportation efficiency and safety, yet ensuring their reliability in complex real-world environments remains a critical challenge. Effective testing is essential to validate ADS performance and reduce deployment risks. This study investigates current ADS testing practices for both modular and end-to-end systems, identifies key demands from industry practitioners and academic researchers, and analyzes the gaps between existing research and real-world requirements. We review major testing techniques and further consider emerging factors such as Vehicle-to-Everything (V2X) communication and foundation models, including large language models and vision foundation models, to understand their roles in enhancing ADS testing. We conducted a large-scale survey with 100 participants from both industry and academia. Survey questions were refined through expert discussions, followed by quantitative and qualitative analyses to reveal key trends, challenges, and unmet needs. Our results show that existing ADS testing techniques struggle to comprehensively evaluate real-world performance, particularly regarding corner case diversity, the simulation to reality gap, the lack of systematic testing criteria, exposure to potential attacks, practical challenges in V2X deployment, and the high computational cost of foundation model-based testing. By further analyzing participant responses together with 105 representative studies, we summarize the current research landscape and highlight major limitations. This study consolidates critical research gaps in ADS testing and outlines key future research directions, including comprehensive testing criteria, cross-model collaboration in V2X systems, cross-modality adaptation for foundation model-based testing, and scalable validation frameworks for large-scale ADS evaluation.

摘要: 自动驾驶系统（ADS）有望提高运输效率和安全性，但确保其在复杂现实环境中的可靠性仍然是一个严峻的挑战。有效的测试对于验证ADS性能和降低部署风险至关重要。本研究调查了模块化和端到端系统的当前ADS测试实践，确定了行业从业者和学术研究人员的关键需求，并分析了现有研究与现实世界要求之间的差距。我们回顾了主要的测试技术，并进一步考虑新兴因素，例如车辆到一切（V2X）通信和基础模型，包括大型语言模型和视觉基础模型，以了解它们在增强ADS测试中的作用。我们对来自工业界和学术界的100名参与者进行了一项大规模调查。通过专家讨论完善了调查问题，然后进行定量和定性分析，以揭示关键趋势、挑战和未满足的需求。我们的结果表明，现有的ADS测试技术很难全面评估现实世界的性能，特别是在角情况多样性、模拟与现实的差距、缺乏系统性测试标准、暴露于潜在攻击、V2X部署中的实际挑战以及基于基础模型的高计算成本方面。测试。通过进一步分析参与者的反应以及105项代表性研究，我们总结了当前的研究格局并强调了主要局限性。本研究巩固了ADS测试中的关键研究空白，并概述了未来的关键研究方向，包括全面的测试标准、V2X系统中的跨模型协作、基于基础模型的测试的跨模式适应，以及用于大规模ADS评估的可扩展验证框架。



## **24. A Practical Framework for Evaluating Medical AI Security: Reproducible Assessment of Jailbreaking and Privacy Vulnerabilities Across Clinical Specialties**

评估医疗人工智能安全性的实用框架：跨临床专业越狱和隐私漏洞的可重复性评估 cs.CR

6 pages, 1 figure, framework proposal

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08185v1) [paper-pdf](https://arxiv.org/pdf/2512.08185v1)

**Authors**: Jinghao Wang, Ping Zhang, Carter Yagemann

**Abstract**: Medical Large Language Models (LLMs) are increasingly deployed for clinical decision support across diverse specialties, yet systematic evaluation of their robustness to adversarial misuse and privacy leakage remains inaccessible to most researchers. Existing security benchmarks require GPU clusters, commercial API access, or protected health data -- barriers that limit community participation in this critical research area. We propose a practical, fully reproducible framework for evaluating medical AI security under realistic resource constraints. Our framework design covers multiple medical specialties stratified by clinical risk -- from high-risk domains such as emergency medicine and psychiatry to general practice -- addressing jailbreaking attacks (role-playing, authority impersonation, multi-turn manipulation) and privacy extraction attacks. All evaluation utilizes synthetic patient records requiring no IRB approval. The framework is designed to run entirely on consumer CPU hardware using freely available models, eliminating cost barriers. We present the framework specification including threat models, data generation methodology, evaluation protocols, and scoring rubrics. This proposal establishes a foundation for comparative security assessment of medical-specialist models and defense mechanisms, advancing the broader goal of ensuring safe and trustworthy medical AI systems.

摘要: 医学大型语言模型（LLM）越来越多地被部署用于不同专业的临床决策支持，但大多数研究人员仍然无法对其对抗性滥用和隐私泄露的稳健性进行系统评估。现有的安全基准测试需要图形处理器集群、商业API访问或受保护的健康数据--这些障碍限制了社区参与这一关键研究领域。我们提出了一个实用、完全可重复的框架，用于在现实资源限制下评估医疗人工智能安全性。我们的框架设计涵盖了按临床风险分层的多个医学专业--从急诊医学和精神病学等高风险领域到全科医学--解决越狱攻击（角色扮演、权威模仿、多回合操纵）和隐私提取攻击。所有评估均使用合成患者记录，无需获得机构审核委员会批准。该框架旨在完全在使用免费型号的消费者中央处理器硬件上运行，消除了成本障碍。我们提出了框架规范，包括威胁模型、数据生成方法、评估协议和评分规则。该提案为医疗专家模型和防御机制的比较安全评估奠定了基础，推进确保医疗人工智能系统安全可信的更广泛目标。



## **25. Detecting Ambiguity Aversion in Cyberattack Behavior to Inform Cognitive Defense Strategies**

检测网络攻击行为中的模糊厌恶以告知认知防御策略 cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.08107v1) [paper-pdf](https://arxiv.org/pdf/2512.08107v1)

**Authors**: Stephan Carney, Soham Hans, Sofia Hirschmann, Stacey Marsella, Yvonne Fonken, Peggy Wu, Nikolos Gurney

**Abstract**: Adversaries (hackers) attempting to infiltrate networks frequently face uncertainty in their operational environments. This research explores the ability to model and detect when they exhibit ambiguity aversion, a cognitive bias reflecting a preference for known (versus unknown) probabilities. We introduce a novel methodological framework that (1) leverages rich, multi-modal data from human-subjects red-team experiments, (2) employs a large language model (LLM) pipeline to parse unstructured logs into MITRE ATT&CK-mapped action sequences, and (3) applies a new computational model to infer an attacker's ambiguity aversion level in near-real time. By operationalizing this cognitive trait, our work provides a foundational component for developing adaptive cognitive defense strategies.

摘要: 试图渗透网络的对手（黑客）经常面临其操作环境的不确定性。这项研究探索了建模和检测他们何时表现出歧义厌恶的能力，这是一种反映对已知（与未知）概率偏好的认知偏见。我们引入了一种新颖的方法论框架，（1）利用来自人类受试者红队实验的丰富、多模式数据，（2）采用大型语言模型（LLM）管道将非结构化日志解析为MITRE ATA和CK映射的动作序列，（3）应用新的计算模型来近实时地推断攻击者的歧义厌恶水平。通过操作这种认知特征，我们的工作为开发适应性认知防御策略提供了基础组成部分。



## **26. RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking of Large Language Models**

RL-MTJail：用于大型语言模型自动黑匣子多回合越狱的强化学习 cs.AI

19 pages, 15 figures

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07761v1) [paper-pdf](https://arxiv.org/pdf/2512.07761v1)

**Authors**: Xiqiao Xiong, Ouxiang Li, Zhuo Liu, Moxin Li, Wentao Shi, Fuli Feng, Xiangnan He

**Abstract**: Large language models are vulnerable to jailbreak attacks, threatening their safe deployment in real-world applications. This paper studies black-box multi-turn jailbreaks, aiming to train attacker LLMs to elicit harmful content from black-box models through a sequence of prompt-output interactions. Existing approaches typically rely on single turn optimization, which is insufficient for learning long-term attack strategies. To bridge this gap, we formulate the problem as a multi-turn reinforcement learning task, directly optimizing the harmfulness of the final-turn output as the outcome reward. To mitigate sparse supervision and promote long-term attack strategies, we propose two heuristic process rewards: (1) controlling the harmfulness of intermediate outputs to prevent triggering the black-box model's rejection mechanisms, and (2) maintaining the semantic relevance of intermediate outputs to avoid drifting into irrelevant content. Experimental results on multiple benchmarks show consistently improved attack success rates across multiple models, highlighting the effectiveness of our approach. The code is available at https://github.com/xxiqiao/RL-MTJail. Warning: This paper contains examples of harmful content.

摘要: 大型语言模型容易受到越狱攻击，威胁其在现实世界应用程序中的安全部署。本文研究黑匣子多回合越狱，旨在训练攻击者LLM通过一系列预算-输出交互从黑匣子模型中获取有害内容。现有的方法通常依赖于单轮优化，这不足以学习长期攻击策略。为了弥合这一差距，我们将问题制定为多回合强化学习任务，直接优化最终回合输出的危害性作为结果奖励。为了减轻稀疏监督并促进长期攻击策略，我们提出了两个启发式过程奖励：（1）控制中间输出的危害性，以防止触发黑匣子模型的拒绝机制，（2）维护中间输出的语义相关性，以避免陷入不相关的内容。多个基准测试的实验结果显示，多个模型的攻击成功率持续提高，凸显了我们方法的有效性。该代码可在https://github.com/xxiqiao/RL-MTJail上获取。警告：本文包含有害内容的示例。



## **27. When Large Language Models Do Not Work: Online Incivility Prediction through Graph Neural Networks**

当大型语言模型不起作用时：通过图神经网络进行在线不文明预测 cs.CL

10 pages

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07684v1) [paper-pdf](https://arxiv.org/pdf/2512.07684v1)

**Authors**: Zihan Chen, Lanyu Yu

**Abstract**: Online incivility has emerged as a widespread and persistent problem in digital communities, imposing substantial social and psychological burdens on users. Although many platforms attempt to curb incivility through moderation and automated detection, the performance of existing approaches often remains limited in both accuracy and efficiency. To address this challenge, we propose a Graph Neural Network (GNN) framework for detecting three types of uncivil behavior (i.e., toxicity, aggression, and personal attacks) within the English Wikipedia community. Our model represents each user comment as a node, with textual similarity between comments defining the edges, allowing the network to jointly learn from both linguistic content and relational structures among comments. We also introduce a dynamically adjusted attention mechanism that adaptively balances nodal and topological features during information aggregation. Empirical evaluations demonstrate that our proposed architecture outperforms 12 state-of-the-art Large Language Models (LLMs) across multiple metrics while requiring significantly lower inference cost. These findings highlight the crucial role of structural context in detecting online incivility and address the limitations of text-only LLM paradigms in behavioral prediction. All datasets and comparative outputs will be publicly available in our repository to support further research and reproducibility.

摘要: 在线不文明已成为数字社区中一个普遍且持续存在的问题，给用户带来了沉重的社会和心理负担。尽管许多平台试图通过审核和自动检测来遏制不文明行为，但现有方法的性能在准确性和效率方面往往仍然有限。为了应对这一挑战，我们提出了一个图形神经网络（GNN）框架来检测三种类型的不文明行为（即毒性、攻击性和人身攻击）在英语维基百科社区中。我们的模型将每个用户评论表示为一个节点，评论之间的文本相似性定义了边缘，允许网络共同从评论之间的语言内容和关系结构中学习。我们还引入了一种动态调整的注意力机制，可以在信息聚合期间自适应地平衡节点和拓扑特征。经验评估表明，我们提出的架构在多个指标上优于12个最先进的大型语言模型（LLM），同时需要显着更低的推理成本。这些发现强调了结构性上下文在检测在线不文明方面的关键作用，并解决了纯文本LLM范式在行为预测中的局限性。所有数据集和比较输出都将在我们的存储库中公开，以支持进一步的研究和重现性。



## **28. Think-Reflect-Revise: A Policy-Guided Reflective Framework for Safety Alignment in Large Vision Language Models**

思考-反思-修订：大视野语言模型中安全一致的政策引导反思框架 cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07141v1) [paper-pdf](https://arxiv.org/pdf/2512.07141v1)

**Authors**: Fenghua Weng, Chaochao Lu, Xia Hu, Wenqi Shao, Wenjie Wang

**Abstract**: As multimodal reasoning improves the overall capabilities of Large Vision Language Models (LVLMs), recent studies have begun to explore safety-oriented reasoning, aiming to enhance safety awareness by analyzing potential safety risks during the reasoning process before generating the final response. Although such approaches improve safety awareness and interpretability, this single-pass think-then-answer paradigm remains vulnerable to contextual or visual jailbreak attacks. This reveals a critical flaw: single-pass reasoning may overlook explicit harmful content in its own output. Our key insight is to exploit this wasted signal through reflection, which can effectively leverage the malicious content revealed in the first-pass reasoning to enable genuine self-correction and prevent unsafe generations. Motivated by this, we propose Think-Reflect-Revise (TRR), a three-stage training framework designed to enhance the safety alignment of LVLMs through policy-guided self-reflection. We first build a Reflective Safety Reasoning (ReSafe) dataset with 5,000 examples that follow a think-reflect-revise process. We then fine-tune the target model using the ReSafe dataset to initialize reflective behavior, and finally reinforce policy-guided reflection through reinforcement learning. Experimental results show that TRR substantially improves the safety performance of LVLMs across both safety-awareness benchmarks and jailbreak attack evaluations, increasing the overall safe response rate from 42.8% to 87.7% on Qwen2.5-VL-7B, while preserving stable performance on general benchmarks such as MMMU and MMStar. The project page is available at https://think-reflect-revise.github.io/.

摘要: 随着多模式推理提高了大视觉语言模型（LVLM）的整体能力，最近的研究开始探索以安全为导向的推理，旨在通过在生成最终响应之前分析推理过程中的潜在安全风险来增强安全意识。尽管此类方法提高了安全意识和可解释性，但这种单程思考然后回答的范式仍然容易受到上下文或视觉越狱攻击。这揭示了一个关键缺陷：单程推理可能会忽视其输出中明显的有害内容。我们的主要见解是通过反射利用这种浪费的信号，这可以有效地利用第一遍推理中揭示的恶意内容，以实现真正的自我纠正并防止不安全的世代。出于此动机，我们提出了思考-反思-修订（TRR），这是一个三阶段培训框架，旨在通过政策引导的自我反思来增强LVLM的安全一致性。我们首先构建一个反思安全推理（ReSafe）数据集，包含5，000个示例，遵循思考-反思-修改过程。然后，我们使用ReSafe数据集微调目标模型以初始化反射行为，并最终通过强化学习加强政策引导的反射。实验结果表明，TRR在安全意识基准和越狱攻击评估中大幅提高了LVLM的安全性能，将Qwen 2.5-BL-7 B的总体安全响应率从42.8%提高到87.7%，同时在MMMU和MMStar等通用基准上保持稳定的性能。该项目页面可访问https://think-reflect-revise.github.io/。



## **29. ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking**

ThinkTrap：通过无限思维对黑匣子LLM服务进行拒绝服务攻击 cs.CR

This version includes the final camera-ready manuscript accepted by NDSS 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07086v1) [paper-pdf](https://arxiv.org/pdf/2512.07086v1)

**Authors**: Yunzhe Li, Jianan Wang, Hongzi Zhu, James Lin, Shan Chang, Minyi Guo

**Abstract**: Large Language Models (LLMs) have become foundational components in a wide range of applications, including natural language understanding and generation, embodied intelligence, and scientific discovery. As their computational requirements continue to grow, these models are increasingly deployed as cloud-based services, allowing users to access powerful LLMs via the Internet. However, this deployment model introduces a new class of threat: denial-of-service (DoS) attacks via unbounded reasoning, where adversaries craft specially designed inputs that cause the model to enter excessively long or infinite generation loops. These attacks can exhaust backend compute resources, degrading or denying service to legitimate users. To mitigate such risks, many LLM providers adopt a closed-source, black-box setting to obscure model internals. In this paper, we propose ThinkTrap, a novel input-space optimization framework for DoS attacks against LLM services even in black-box environments. The core idea of ThinkTrap is to first map discrete tokens into a continuous embedding space, then undertake efficient black-box optimization in a low-dimensional subspace exploiting input sparsity. The goal of this optimization is to identify adversarial prompts that induce extended or non-terminating generation across several state-of-the-art LLMs, achieving DoS with minimal token overhead. We evaluate the proposed attack across multiple commercial, closed-source LLM services. Our results demonstrate that, even far under the restrictive request frequency limits commonly enforced by these platforms, typically capped at ten requests per minute (10 RPM), the attack can degrade service throughput to as low as 1% of its original capacity, and in some cases, induce complete service failure.

摘要: 大型语言模型（LLM）已经成为广泛应用的基础组件，包括自然语言理解和生成，体现智能和科学发现。随着计算需求的不断增长，这些模型越来越多地部署为基于云的服务，允许用户通过互联网访问强大的LLM。然而，这种部署模型引入了一类新的威胁：通过无限推理的拒绝服务（DoS）攻击，其中攻击者精心设计了特别设计的输入，导致模型进入过长或无限的生成循环。这些攻击可能耗尽后端计算资源，降低或拒绝向合法用户提供服务。为了降低此类风险，许多LLM提供商采用闭源、黑匣子设置来掩盖模型内部内容。在本文中，我们提出了ThinkTrap，这是一种新型的输入空间优化框架，即使在黑匣子环境中也可以对LLM服务进行NOS攻击。ThinkTrap的核心思想是首先将离散令牌映射到连续嵌入空间，然后在利用输入稀疏性的低维子空间中进行高效的黑匣子优化。此优化的目标是识别在多个最先进的LLM之间引发扩展或非终止生成的对抗提示，以最小的令牌负载实现拒绝服务。我们评估了跨多个商业、闭源LLM服务的拟议攻击。我们的结果表明，即使远低于这些平台通常强制执行的限制性请求频率限制（通常限制为每分钟10个请求（10转/分钟）），攻击也可以将服务吞吐量降低至低至其原始容量的1%，在某些情况下，会导致完全的服务故障。



## **30. Replicating TEMPEST at Scale: Multi-Turn Adversarial Attacks Against Trillion-Parameter Frontier Models**

大规模复制TEMPST：针对万亿参数前沿模型的多轮对抗攻击 cs.CL

30 pages, 11 figures, 5 tables. Code and data: https://github.com/ricyoung/tempest-replication

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07059v1) [paper-pdf](https://arxiv.org/pdf/2512.07059v1)

**Authors**: Richard Young

**Abstract**: Despite substantial investment in safety alignment, the vulnerability of large language models to sophisticated multi-turn adversarial attacks remains poorly characterized, and whether model scale or inference mode affects robustness is unknown. This study employed the TEMPEST multi-turn attack framework to evaluate ten frontier models from eight vendors across 1,000 harmful behaviors, generating over 97,000 API queries across adversarial conversations with automated evaluation by independent safety classifiers. Results demonstrated a spectrum of vulnerability: six models achieved 96% to 100% attack success rate (ASR), while four showed meaningful resistance, with ASR ranging from 42% to 78%; enabling extended reasoning on identical architecture reduced ASR from 97% to 42%. These findings indicate that safety alignment quality varies substantially across vendors, that model scale does not predict adversarial robustness, and that thinking mode provides a deployable safety enhancement. Collectively, this work establishes that current alignment techniques remain fundamentally vulnerable to adaptive multi-turn attacks regardless of model scale, while identifying deliberative inference as a promising defense direction.

摘要: 尽管在安全对齐方面投入了大量资金，但大型语言模型对复杂多轮对抗攻击的脆弱性仍然很难描述，并且模型规模或推理模式是否会影响稳健性尚不清楚。这项研究采用TEMPEST多回合攻击框架来评估来自8家供应商的10个前沿模型，涵盖1，000种有害行为，通过独立安全分类器的自动评估，在对抗性对话中生成超过97，000个API查询。结果展示了一系列脆弱性：六个模型实现了96%至100%的攻击成功率（ASB），四个模型表现出有意义的抵抗力，ASB范围从42%至78%;在相同的架构上启用扩展推理将ASC从97%降低到42%。这些发现表明，不同供应商的安全对齐质量存在很大差异，模型规模无法预测对抗稳健性，并且思维模式提供了可部署的安全增强。总的来说，这项工作确定了，无论模型规模如何，当前的对齐技术仍然从根本上容易受到自适应多转弯攻击的影响，同时将刻意推理确定为一个有希望的防御方向。



## **31. SoK: Trust-Authorization Mismatch in LLM Agent Interactions**

SoK：LLM代理交互中的信任授权不匹配 cs.CR

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06914v1) [paper-pdf](https://arxiv.org/pdf/2512.06914v1)

**Authors**: Guanquan Shi, Haohua Du, Zhiqiang Wang, Xiaoyu Liang, Weiwenpei Liu, Song Bian, Zhenyu Guan

**Abstract**: Large Language Models (LLMs) are rapidly evolving into autonomous agents capable of interacting with the external world, significantly expanding their capabilities through standardized interaction protocols. However, this paradigm revives the classic cybersecurity challenges of agency and authorization in a novel and volatile context. As decision-making shifts from deterministic code logic to probabilistic inference driven by natural language, traditional security mechanisms designed for deterministic behavior fail. It is fundamentally challenging to establish trust for unpredictable AI agents and to enforce the Principle of Least Privilege (PoLP) when instructions are ambiguous. Despite the escalating threat landscape, the academic community's understanding of this emerging domain remains fragmented, lacking a systematic framework to analyze its root causes. This paper provides a unifying formal lens for agent-interaction security.   We observed that most security threats in this domain stem from a fundamental mismatch between trust evaluation and authorization policies. We introduce a novel risk analysis model centered on this trust-authorization gap. Using this model as a unifying lens, we survey and classify the implementation paths of existing, often seemingly isolated, attacks and defenses. This new framework not only unifies the field but also allows us to identify critical research gaps. Finally, we leverage our analysis to suggest a systematic research direction toward building robust, trusted agents and dynamic authorization mechanisms.

摘要: 大型语言模型（LLM）正在迅速演变为能够与外部世界交互的自治代理，通过标准化的交互协议显着扩展其能力。然而，这种范式在新颖且不稳定的背景下重新焕发了代理和授权的经典网络安全挑战。随着决策从确定性代码逻辑转向自然语言驱动的概率推理，为确定性行为设计的传统安全机制就会失败。为不可预测的人工智能代理建立信任并在指令模糊时执行最小特权原则（PoLP）是一个根本性的挑战。尽管威胁格局不断升级，但学术界对这一新兴领域的理解仍然支离破碎，缺乏系统性的框架来分析其根本原因。本文为代理交互安全性提供了统一的正式视角。   我们观察到，该领域中的大多数安全威胁源于信任评估和授权策略之间的根本不匹配。我们以这种信任-授权差距为中心引入了一种新颖的风险分析模型。使用这个模型作为统一的镜头，我们调查和分类了现有的（通常看似孤立的）攻击和防御的实施路径。这个新框架不仅统一了该领域，还使我们能够识别关键的研究差距。最后，我们利用我们的分析提出了一个系统性的研究方向，以构建稳健、可信的代理和动态授权机制。



## **32. From Description to Score: Can LLMs Quantify Vulnerabilities?**

从描述到评分：LLM可以量化漏洞吗？ cs.CR

10 pages

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06781v1) [paper-pdf](https://arxiv.org/pdf/2512.06781v1)

**Authors**: Sima Jafarikhah, Daniel Thompson, Eva Deans, Hossein Siadati, Yi Liu

**Abstract**: Manual vulnerability scoring, such as assigning Common Vulnerability Scoring System (CVSS) scores, is a resource-intensive process that is often influenced by subjective interpretation. This study investigates the potential of general-purpose large language models (LLMs), namely ChatGPT, Llama, Grok, DeepSeek, and Gemini, to automate this process by analyzing over 31{,}000 recent Common Vulnerabilities and Exposures (CVE) entries. The results show that LLMs substantially outperform the baseline on certain metrics (e.g., \textit{Availability Impact}), while offering more modest gains on others (e.g., \textit{Attack Complexity}). Moreover, model performance varies across both LLM families and individual CVSS metrics, with ChatGPT-5 attaining the highest precision. Our analysis reveals that LLMs tend to misclassify many of the same CVEs, and ensemble-based meta-classifiers only marginally improve performance. Further examination shows that CVE descriptions often lack critical context or contain ambiguous phrasing, which contributes to systematic misclassifications. These findings underscore the importance of enhancing vulnerability descriptions and incorporating richer contextual details to support more reliable automated reasoning and alleviate the growing backlog of CVEs awaiting triage.

摘要: 手动漏洞评分，例如分配通用漏洞评分系统（CVD）分数，是一个资源密集型的过程，通常受到主观解释的影响。本研究调查了通用大型语言模型（LLM）（即ChatGPT、Llama、Grok、DeepSeek和Gemini）的潜力，通过分析超过31{，}000个最近的常见漏洞和暴露（UTE）条目来自动化该过程。结果表明，LLM在某些指标上的表现大大优于基线（例如，\textit{可用性影响}），同时为其他人提供更适度的收益（例如，\textit{Attack Complexity}）。此外，模型性能因LLM系列和单个CVD指标而异，其中ChatGPT-5达到了最高的精确度。我们的分析表明，LLM往往会对许多相同的CVE进行错误分类，而基于集成的元分类器只能略微提高性能。进一步的检查表明，UTE描述通常缺乏关键上下文或包含模棱两可的措辞，这导致了系统性的错误分类。这些发现强调了增强漏洞描述和整合更丰富的上下文细节的重要性，以支持更可靠的自动化推理并减轻等待分诊的CVS不断增加的积压。



## **33. VRSA: Jailbreaking Multimodal Large Language Models through Visual Reasoning Sequential Attack**

VRSA：通过视觉推理序列攻击破解多模式大型语言模型 cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.05853v2) [paper-pdf](https://arxiv.org/pdf/2512.05853v2)

**Authors**: Shiji Zhao, Shukun Xiong, Yao Huang, Yan Jin, Zhenyu Wu, Jiyang Guan, Ranjie Duan, Jialing Tao, Hui Xue, Xingxing Wei

**Abstract**: Multimodal Large Language Models (MLLMs) are widely used in various fields due to their powerful cross-modal comprehension and generation capabilities. However, more modalities bring more vulnerabilities to being utilized for jailbreak attacks, which induces MLLMs to output harmful content. Due to the strong reasoning ability of MLLMs, previous jailbreak attacks try to explore reasoning safety risk in text modal, while similar threats have been largely overlooked in the visual modal. To fully evaluate potential safety risks in the visual reasoning task, we propose Visual Reasoning Sequential Attack (VRSA), which induces MLLMs to gradually externalize and aggregate complete harmful intent by decomposing the original harmful text into several sequentially related sub-images. In particular, to enhance the rationality of the scene in the image sequence, we propose Adaptive Scene Refinement to optimize the scene most relevant to the original harmful query. To ensure the semantic continuity of the generated image, we propose Semantic Coherent Completion to iteratively rewrite each sub-text combined with contextual information in this scene. In addition, we propose Text-Image Consistency Alignment to keep the semantical consistency. A series of experiments demonstrates that the VRSA can achieve a higher attack success rate compared with the state-of-the-art jailbreak attack methods on both the open-source and closed-source MLLMs such as GPT-4o and Claude-4.5-Sonnet.

摘要: 多模式大型语言模型（MLLM）因其强大的跨模式理解和生成能力而被广泛应用于各个领域。然而，更多的模式会带来更多被用于越狱攻击的漏洞，从而导致MLLM输出有害内容。由于MLLM推理能力强，之前的越狱攻击试图在文本模式中探索推理安全风险，而类似的威胁在视觉模式中基本上被忽视。为了充分评估视觉推理任务中潜在的安全风险，我们提出了视觉推理序列攻击（VRSA），通过将原始有害文本分解为几个顺序相关的子图像，诱导MLLM逐渐外部化和聚合完整的有害意图。特别是，为了增强图像序列中场景的合理性，我们提出自适应场景细化来优化与原始有害查询最相关的场景。为了确保生成图像的语义连续性，我们提出了语义连贯完成来迭代重写该场景中结合上下文信息的每个子文本。此外，我们还提出了文本-图像一致性对齐来保持语义一致性。一系列实验表明，与最先进的越狱攻击方法相比，VRSA可以在GPT-4 o和Claude-4.5-Sonnet等开源和闭源MLLM上实现更高的攻击成功率。



## **34. TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations**

TeleAI-Safety：针对攻击、防御和评估的全面LLM越狱基准 cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.05485v2) [paper-pdf](https://arxiv.org/pdf/2512.05485v2)

**Authors**: Xiuyuan Chen, Jian Zhao, Yuxiang He, Yuan Xun, Xinwei Liu, Yanshu Li, Huilin Zhou, Wei Cai, Ziyan Shi, Yuchen Yuan, Tianle Zhang, Chi Zhang, Xuelong Li

**Abstract**: While the deployment of large language models (LLMs) in high-value industries continues to expand, the systematic assessment of their safety against jailbreak and prompt-based attacks remains insufficient. Existing safety evaluation benchmarks and frameworks are often limited by an imbalanced integration of core components (attack, defense, and evaluation methods) and an isolation between flexible evaluation frameworks and standardized benchmarking capabilities. These limitations hinder reliable cross-study comparisons and create unnecessary overhead for comprehensive risk assessment. To address these gaps, we present TeleAI-Safety, a modular and reproducible framework coupled with a systematic benchmark for rigorous LLM safety evaluation. Our framework integrates a broad collection of 19 attack methods (including one self-developed method), 29 defense methods, and 19 evaluation methods (including one self-developed method). With a curated attack corpus of 342 samples spanning 12 distinct risk categories, the TeleAI-Safety benchmark conducts extensive evaluations across 14 target models. The results reveal systematic vulnerabilities and model-specific failure cases, highlighting critical trade-offs between safety and utility, and identifying potential defense patterns for future optimization. In practical scenarios, TeleAI-Safety can be flexibly adjusted with customized attack, defense, and evaluation combinations to meet specific demands. We release our complete code and evaluation results to facilitate reproducible research and establish unified safety baselines.

摘要: 尽管大型语言模型（LLM）在高价值行业的部署持续扩大，但对其针对越狱和预算攻击的安全性的系统评估仍然不足。现有的安全评估基准和框架通常受到核心组件（攻击、防御和评估方法）的不平衡集成以及灵活评估框架和标准化基准能力之间的隔离的限制。这些限制阻碍了可靠的交叉研究比较，并为全面风险评估带来了不必要的费用。为了解决这些差距，我们提出了TeleAI-Safety，这是一个模块化、可重复的框架，结合了严格LLM安全评估的系统基准。我们的框架集成了19种攻击方法（包括一种自主开发的方法），29种防御方法和19种评估方法（包括一种自主开发的方法）。TeleAI-Safety基准测试使用了涵盖12个不同风险类别的342个样本的策划攻击语料库，对14个目标模型进行了广泛的评估。结果揭示了系统漏洞和特定于模型的故障案例，突出了安全性和实用性之间的关键权衡，并确定了未来优化的潜在防御模式。在实际场景中，TeleAI-Safety可以灵活调整自定义攻击、防御和评估组合，以满足特定需求。我们发布完整的代码和评估结果，以促进可重复的研究并建立统一的安全基线。



## **35. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

当对齐失败时：对视觉-语言-动作模型的多模式对抗攻击 cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2511.16203v3) [paper-pdf](https://arxiv.org/pdf/2511.16203v3)

**Authors**: Yuping Yan, Yuhan Xie, Yixin Zhang, Lingjuan Lyu, Handing Wang, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.

摘要: 视觉-语言-动作模型（VLA）最近在具体环境中取得了显着进展，使机器人能够通过统一的多模式理解来感知、推理和行动。尽管它们的能力令人印象深刻，但这些系统的对抗鲁棒性在很大程度上仍未得到探索，尤其是在现实的多模式和黑匣子条件下。现有的研究主要关注单模式扰动，而忽视了从根本上影响体现推理和决策的跨模式失调。本文介绍了VLA-Fool，这是对白盒和黑盒设置下具体VLA模型中多模式对抗鲁棒性的全面研究。VLA-Fool统一了三个级别的多模式对抗攻击：（1）通过基于梯度和基于预算的操纵进行文本扰动，（2）通过补丁和噪音失真进行视觉扰动，以及（3）故意破坏感知和指令之间的语义对应性的跨模式失准攻击。我们进一步将VLA感知的语义空间融入到语言提示中，开发了第一个自动制作和语义引导的提示框架。使用微调的OpenVLA模型对LIBERO基准进行的实验表明，即使是微小的多峰扰动也会导致显着的行为偏差，这表明了体现多峰对齐的脆弱性。



## **36. Tight and Practical Privacy Auditing for Differentially Private In-Context Learning**

针对差异私密的上下文学习进行严格而实用的隐私审计 cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.13502v2) [paper-pdf](https://arxiv.org/pdf/2511.13502v2)

**Authors**: Yuyang Xia, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) perform in-context learning (ICL) by adapting to tasks from prompt demonstrations, which in practice often contain private or proprietary data. Although differential privacy (DP) with private voting is a pragmatic mitigation, DP-ICL implementations are error-prone, and worst-case DP bounds may substantially overestimate actual leakage, calling for practical auditing tools. We present a tight and efficient privacy auditing framework for DP-ICL systems that runs membership inference attacks and translates their success rates into empirical privacy guarantees using Gaussian DP. Our analysis of the private voting mechanism identifies vote configurations that maximize the auditing signal, guiding the design of audit queries that reliably reveal whether a canary demonstration is present in the context. The framework supports both black-box (API-only) and white-box (internal vote) threat models, and unifies auditing for classification and generation by reducing both to a binary decision problem. Experiments on standard text classification and generation benchmarks show that our empirical leakage estimates closely match theoretical DP budgets on classification tasks and are consistently lower on generation tasks due to conservative embedding-sensitivity bounds, making our framework a practical privacy auditor and verifier for real-world DP-ICL deployments.

摘要: 大型语言模型（LLM）通过适应即时演示的任务来执行上下文学习（ICL），这些任务在实践中通常包含私人或专有数据。尽管带有私人投票的差异隐私（DP）是一种务实的缓解措施，但DP-ICL的实现很容易出错，而且最坏情况下的DP界限可能会大大高估实际泄漏，因此需要实用的审计工具。我们为DP-ICL系统提供了一个严格而高效的隐私审计框架，该框架运行成员资格推断攻击，并使用高斯DP将其成功率转化为经验隐私保证。我们对私人投票机制的分析确定了最大化审计信号的投票配置，指导审计查询的设计，可靠地揭示上下文中是否存在金丝雀演示。该框架支持黑匣子（仅API）和白盒（内部投票）威胁模型，并通过将两者简化为二元决策问题来统一分类和生成审计。标准文本分类和生成基准的实验表明，我们的经验泄露估计与分类任务的理论DP预算密切匹配，并且由于保守的嵌入敏感性界限，生成任务的泄漏估计始终较低，使我们的框架成为现实世界DP-ICL部署的实用隐私审计器和验证器。



## **37. Verifying LLM Inference to Detect Model Weight Exfiltration**

CLARLLM推理检测模型重量溢出 cs.CR

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2511.02620v2) [paper-pdf](https://arxiv.org/pdf/2511.02620v2)

**Authors**: Roy Rinberg, Adam Karvonen, Alexander Hoover, Daniel Reuter, Keri Warr

**Abstract**: As large AI models become increasingly valuable assets, the risk of model weight exfiltration from inference servers grows accordingly. An attacker controlling an inference server may exfiltrate model weights by hiding them within ordinary model outputs, a strategy known as steganography. This work investigates how to verify model responses to defend against such attacks and, more broadly, to detect anomalous or buggy behavior during inference. We formalize model exfiltration as a security game, propose a verification framework that can provably mitigate steganographic exfiltration, and specify the trust assumptions associated with our scheme. To enable verification, we characterize valid sources of non-determinism in large language model inference and introduce two practical estimators for them. We evaluate our detection framework on several open-weight models ranging from 3B to 30B parameters. On MOE-Qwen-30B, our detector reduces exfiltratable information to <0.5% with false-positive rate of 0.01%, corresponding to a >200x slowdown for adversaries. Overall, this work further establishes a foundation for defending against model weight exfiltration and demonstrates that strong protection can be achieved with minimal additional cost to inference providers.

摘要: 随着大型人工智能模型成为越来越有价值的资产，模型权重从推理服务器泄露的风险也相应增加。控制推理服务器的攻击者可以通过将模型权重隐藏在普通模型输出中来提取模型权重，这种策略称为隐写术。这项工作研究了如何验证模型响应以抵御此类攻击，以及更广泛地说，在推理过程中检测异常或有缺陷的行为。我们将模型溢出形式化为一个安全游戏，提出了一个可以证明减轻隐写溢出的验证框架，并指定与我们的方案相关的信任假设。为了实现验证，我们描述了大型语言模型推理中非决定性的有效来源，并为其引入了两个实用的估计器。我们在从3B到30 B参数的几个开放权重模型上评估了我们的检测框架。在MOE-Qwen-30 B上，我们的检测器将可渗透信息减少到<0.5%，假阳性率为0.01%，相当于对手的速度减慢> 200倍。总体而言，这项工作进一步奠定了防御模型权重溢出的基础，并证明可以以最小的额外成本来实现强大的保护。



## **38. BreakFun: Jailbreaking LLMs via Schema Exploitation**

BreakFun：通过模式利用越狱LLM cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2510.17904v2) [paper-pdf](https://arxiv.org/pdf/2510.17904v2)

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.

摘要: 大型语言模型（LLM）在处理结构化数据和遵守语法规则方面的熟练程度是推动其广泛采用的一种能力，但也使它们变得脆弱。在本文中，我们通过BreakFun调查了这个漏洞，BreakFun是一种越狱方法，可以将LLM对结构化模式的遵守武器化。BreakFun采用了一个由三部分组成的提示，将一个无辜的框架和一个思想链分散注意力与一个核心“特洛伊模式”相结合-一个精心制作的数据结构，迫使模型生成有害内容，利用LLM遵循结构和模式的强烈倾向。我们证明该漏洞具有高度可转移性，JailbreakBench上的13个基础和专有模型平均成功率为89%，并在几个著名模型上达到100%的攻击成功率（ASB）。一项严格的消融研究证实，该特洛伊模式是攻击的主要原因。为了解决这个问题，我们引入了对抗性提示解构护栏，这是一种利用二级LLM来执行“文字转录”的防御--提取所有人类可读的文本以隔离和揭示用户真正的有害意图。我们的概念验证护栏展示了针对攻击的高功效，验证了针对欺骗性模式是一种可行的缓解策略。我们的工作探讨了法学硕士的核心优势如何转化为关键弱点，为构建更稳健一致的模型提供了新的视角。



## **39. PEAR: Planner-Executor Agent Robustness Benchmark**

PEAR：规划者-执行者代理稳健性基准 cs.LG

arXiv admin note: This submission has been withdrawn by arXiv administrators due to incorrect authorship. Author list truncated

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2510.07505v3) [paper-pdf](https://arxiv.org/pdf/2510.07505v3)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）已成为处理跨不同领域复杂、多步骤任务的强大范式。然而，尽管MAS的能力令人印象深刻，但仍然容易受到对抗操纵。现有的研究通常会检查孤立的攻击表面或特定场景，从而缺乏对MAS漏洞的全面了解。为了弥合这一差距，我们引入了PEAR，这是一个用于系统评估规划者-执行者MAS的实用性和脆弱性的基准。虽然与各种MAS架构兼容，但我们的基准测试重点关注规划者-执行者结构，这是一种实用且广泛采用的设计。通过大量实验，我们发现（1）弱规划者比弱执行者更严重地降低总体清洁任务性能;（2）虽然内存模块对于规划者来说至关重要，但为执行者配备内存模块并不会影响清洁任务性能;（3）任务性能和鲁棒性之间存在权衡;以及（4）针对计划者的攻击对于误导系统特别有效。这些发现为增强MAS的稳健性提供了可行的见解，并为多代理环境中的原则性防御奠定了基础。



## **40. A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks**

针对即时注入攻击的多代理LLM防御管道 cs.CR

Accepted at the 11th IEEE WIECON-ECE 2025

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2509.14285v3) [paper-pdf](https://arxiv.org/pdf/2509.14285v3)

**Authors**: S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin

**Abstract**: Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.

摘要: 提示注入攻击是大型语言模型（LLM）部署中的一个主要漏洞，用户输入中嵌入的恶意指令可以覆盖系统提示并引发意外行为。本文提出了一种新型的多代理防御框架，该框架在协调管道中使用专门的LLM代理来实时检测和抵消即时注入攻击。我们使用两种不同的架构来评估我们的方法：顺序代理链管道和基于分层协调器的系统。我们对两个LLM平台（ChatGLM和Llama 2）上的55种独特的即时注入攻击（分为8类，总共400个攻击实例）进行了全面评估，展示了显着的安全改进。在没有防御机制的情况下，ChatGLM的基线攻击成功率（ASB）达到30%，Llama 2的基线攻击成功率（ASB）达到20%。我们的多代理管道实现了100%的缓解，在所有测试场景中将ASB降低至0%。该框架展示了多种攻击类别的稳健性，包括直接覆盖、代码执行尝试、数据溢出和模糊技术，同时维护合法查询的系统功能。



## **41. Unveiling the Latent Directions of Reflection in Large Language Models**

揭示大型语言模型中反射的潜在方向 cs.LG

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2508.16989v2) [paper-pdf](https://arxiv.org/pdf/2508.16989v2)

**Authors**: Fu-Chieh Chang, Yu-Ting Lee, Pei-Yuan Wu

**Abstract**: Reflection, the ability of large language models (LLMs) to evaluate and revise their own reasoning, has been widely used to improve performance on complex reasoning tasks. Yet, most prior works emphasizes designing reflective prompting strategies or reinforcement learning objectives, leaving the inner mechanisms of reflection underexplored. In this paper, we investigate reflection through the lens of latent directions in model activations. We propose a methodology based on activation steering to characterize how instructions with different reflective intentions: no reflection, intrinsic reflection, and triggered reflection. By constructing steering vectors between these reflection levels, we demonstrate that (1) new reflection-inducing instructions can be systematically identified, (2) reflective behavior can be directly enhanced or suppressed through activation interventions, and (3) suppressing reflection is considerably easier than stimulating it. Experiments on GSM8k-adv and Cruxeval-o-adv with Qwen2.5-3B and Gemma3-4B-IT reveal clear stratification across reflection levels, and steering interventions confirm the controllability of reflection. Our findings highlight both opportunities (e.g., reflection-enhancing defenses) and risks (e.g., adversarial inhibition of reflection in jailbreak attacks). This work opens a path toward mechanistic understanding of reflective reasoning in LLMs.

摘要: 反射是大型语言模型（LLM）评估和修改自身推理的能力，已被广泛用于提高复杂推理任务的性能。然而，大多数先前的作品都强调设计反思性提示策略或强化学习目标，而反思的内部机制却没有得到充分的探索。本文中，我们研究了模型激活中潜在方向的镜头的反射。我们提出了一种基于激活引导的方法论来描述具有不同反射意图的指令：无反射、内在反射和触发反射。通过在这些反射水平之间构建引导载体，我们证明了（1）可以系统地识别新的反射诱导指令，（2）可以通过激活干预直接增强或抑制反射行为，和（3）抑制反射比刺激反射容易得多。使用Qwen 2.5 -3B和Gemma 3 - 4 B-对GSM 8 k-adv和Cruxeval-o-adv进行实验它揭示了反射水平之间的明确分层，引导干预证实了反射的可控性。我们的调查结果强调了这两种机会（例如，反思增强防御）和风险（例如，越狱攻击中反思的对抗性抑制）。这项工作开辟了对法学硕士中反思推理的机械理解的道路。



## **42. ConceptGuard: Neuro-Symbolic Safety Guardrails via Sparse Interpretable Jailbreak Concepts**

ConceptGuard：通过稀疏可解释越狱概念的神经符号安全护栏 cs.CL

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2508.16325v2) [paper-pdf](https://arxiv.org/pdf/2508.16325v2)

**Authors**: Darpan Aswal, Céline Hudelot

**Abstract**: Large Language Models have found success in a variety of applications. However, their safety remains a concern due to the existence of various jailbreaking methods. Despite significant efforts, alignment and safety fine-tuning only provide a certain degree of robustness against jailbreak attacks that covertly mislead LLMs towards the generation of harmful content. This leaves them prone to a range of vulnerabilities, including targeted misuse and accidental user profiling. This work introduces \textbf{ConceptGuard}, a novel framework that leverages Sparse Autoencoders (SAEs) to identify interpretable concepts within LLM internals associated with different jailbreak themes. By extracting semantically meaningful internal representations, ConceptGuard enables building robust safety guardrails -- offering fully explainable and generalizable defenses without sacrificing model capabilities or requiring further fine-tuning. Leveraging advances in the mechanistic interpretability of LLMs, our approach provides evidence for a shared activation geometry for jailbreak attacks in the representation space, a potential foundation for designing more interpretable and generalizable safeguards against attackers.

摘要: 大型语言模型在各种应用中取得了成功。然而，由于各种越狱方法的存在，他们的安全仍然是一个问题。尽管做出了巨大的努力，但对齐和安全微调只能对越狱攻击提供一定程度的鲁棒性，这些攻击秘密误导LLM产生有害内容。这使得它们容易受到一系列漏洞，包括有针对性的滥用和意外的用户分析。这项工作引入了\textBF{ConceptGuard}，这是一个新颖的框架，它利用稀疏自动编码器（SAEs）来识别与不同越狱主题相关的LLM内部中的可解释概念。通过提取具有语义意义的内部表示，ConceptGuard能够构建强大的安全护栏--在不牺牲模型能力或需要进一步微调的情况下提供完全可解释和可概括的防御。利用LLM机械可解释性的进步，我们的方法为表示空间中越狱攻击的共享激活几何提供了证据，这是设计针对攻击者的更具可解释性和可概括性的防护措施的潜在基础。



## **43. Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**

缓存中的影子：在LLM推理中揭示和减轻KV缓存的隐私风险 cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2508.09442v3) [paper-pdf](https://arxiv.org/pdf/2508.09442v3)

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.

摘要: Key-Value（KV）缓存存储中间注意力计算（Key和Value对）以避免冗余计算，是加速大型语言模型（LLM）推理的基本机制。然而，这种效率优化引入了重大但未充分探索的隐私风险。本文首次对这些漏洞进行了全面分析，证明攻击者可以直接从KV缓存重建敏感用户输入。我们设计并实现了三种不同的攻击载体：直接倒置攻击、更广泛适用且更强大的碰撞攻击以及基于语义的注入攻击。这些方法证明了KV缓存隐私泄露问题的实用性和严重性。为了缓解这个问题，我们提出了KV-Cloak，这是一种新颖、轻量级且高效的防御机制。KV-Cloak使用基于可逆矩阵的混淆方案，结合操作符融合来保护KV-缓存。我们广泛的实验表明，KV-Cloak有效地阻止了所有提出的攻击，降低了随机噪音的重建质量。至关重要的是，它实现了这种强大的安全性，模型准确性几乎没有下降，性能负担最小，为值得信赖的LLM部署提供了实用的解决方案。



## **44. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

从即时注射到协议漏洞：LLM-Powered AI代理工作流中的威胁 cs.CR

The paper is published in ICT Express (Elsevier)

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2506.23260v2) [paper-pdf](https://arxiv.org/pdf/2506.23260v2)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Abderrahmane Lakas, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces enable real-time data retrieval, computation, and multi-step orchestration. However, the rapid growth of plugins, connectors, and inter-agent protocols has outpaced security practices, leading to brittle integrations that rely on ad-hoc authentication, inconsistent schemas, and weak validation. This survey introduces a unified end-to-end threat model for LLM-agent ecosystems, covering host-to-tool and agent-to-agent communications. We systematically categorize more than thirty attack techniques spanning input manipulation, model compromise, system and privacy attacks, and protocol-level vulnerabilities. For each category, we provide a formal threat formulation defining attacker capabilities, objectives, and affected system layers. Representative examples include Prompt-to-SQL injections and the Toxic Agent Flow exploit in GitHub MCP servers. We analyze attack feasibility, review existing defenses, and discuss mitigation strategies such as dynamic trust management, cryptographic provenance tracking, and sandboxed agent interfaces. The framework is validated through expert review and cross-mapping with real-world incidents and public vulnerability repositories, including CVE and NIST NVD. Compared to prior surveys, this work presents the first integrated taxonomy bridging input-level exploits and protocol-layer vulnerabilities in LLM-agent ecosystems, offering actionable guidance for designing secure and resilient agentic AI systems.

摘要: 自主人工智能代理由大型语言模型（LLM）提供支持，具有结构化功能调用接口，可实现实时数据检索、计算和多步骤编排。然而，插件、连接器和代理间协议的快速发展已经超过了安全实践，导致依赖于临时身份验证、不一致的模式和弱验证的脆弱集成。本调查为LLM代理生态系统引入了统一的端到端威胁模型，涵盖主机到工具和代理到代理的通信。我们系统地分类了三十多种攻击技术，涵盖输入操纵、模型妥协、系统和隐私攻击以及协议级漏洞。对于每个类别，我们提供了定义攻击者能力、目标和受影响的系统层的正式威胁公式。代表性示例包括GitHub LCP服务器中的预算到SQL注入和Toxic Agent Flow漏洞利用。我们分析攻击的可行性，审查现有的防御措施，并讨论动态信任管理、加密出处跟踪和沙箱代理接口等缓解策略。该框架通过专家审查和与现实世界事件和公共漏洞存储库（包括UTE和NIH NVD）的交叉映射进行验证。与之前的调查相比，这项工作提出了第一个集成的分类法，弥合了LLM代理生态系统中的输入级漏洞和协议层漏洞，为设计安全且有弹性的代理人工智能系统提供了可操作的指导。



## **45. OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Languages and Modalities**

OMNIGUARD：跨语言和模态的AI安全适度的有效方法 cs.CL

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2505.23856v2) [paper-pdf](https://arxiv.org/pdf/2505.23856v2)

**Authors**: Sahil Verma, Keegan Hines, Jeff Bilmes, Charlotte Siska, Luke Zettlemoyer, Hila Gonen, Chandan Singh

**Abstract**: The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose Omniguard, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. Omniguard improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, Omniguard is also very efficient ($\approx\!120 \times$ faster than the next fastest baseline). Code and data are available at: https://github.com/vsahil/OmniGuard.

摘要: 大型语言模型（LLM）的新兴功能引发了人们对其直接潜在有害滥用的担忧。缓解这些担忧的核心方法是检测对模型的有害查询。当前的检测方法是容易出错的，并且特别容易受到利用模型能力不匹配的概括的攻击（例如，低资源语言的提示或以图像和音频等非文本形式提供的提示）。为了应对这一挑战，我们提出了Omniguard，一种用于检测跨语言和模式的有害提示的方法。我们的方法（i）识别跨语言或模式对齐的LLM/MLLM的内部表示，然后（ii）使用它们来构建语言不可知或模式不可知的分类器，用于检测有害提示。Omniguard将有害提示分类准确性提高了11.57\%，超过了多语言设置中最强的基线，对于基于图像的提示提高了20.44\%，并为基于音频的提示设置了新的SOTA。通过重新利用在生成过程中计算的嵌入，Omniguard也非常高效（$\approx\！比下一个最快的基线快120倍）。代码和数据可访问：https://github.com/vsahil/OmniGuard。



## **46. CachePrune: Neural-Based Attribution Defense Against Indirect Prompt Injection Attacks**

Cache Prune：针对即时间接注入攻击的基于神经的归因防御 cs.CR

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2504.21228v2) [paper-pdf](https://arxiv.org/pdf/2504.21228v2)

**Authors**: Rui Wang, Junda Wu, Yu Xia, Tong Yu, Ruiyi Zhang, Ryan Rossi, Subrata Mitra, Lina Yao, Julian McAuley

**Abstract**: Large Language Models (LLMs) are susceptible to indirect prompt injection attacks, in which the model inadvertently responds to task messages injected within the prompt context. This vulnerability stems from LLMs' inability to distinguish between data and instructions within a prompt. In this paper, we propose CachePrune, a defense method that identifies and prunes task-triggering neurons from the KV cache of the input prompt context. By pruning such neurons, we encourage the LLM to interpret the input prompt context purely as data rather than as cues for instruction following. To identify these neurons, we introduce a neural attribution mechanism guided by a preferential attribution loss, which enables effective attribution with only a few samples while preserving response quality after pruning. We further enhance the efficacy of neural attribution by leveraging an observed triggering effect inherent in the model's response generation behavior. Notably, our approach does not impose additional formatting on the prompt or introduce extra test-time LLM calls. Experiments show that CachePrune can significantly reduce attack success rates while maintaining clean response quality.

摘要: 大型语言模型（LLM）容易受到间接提示注入攻击，其中模型无意中响应了提示上下文中注入的任务消息。此漏洞源于LLM无法区分提示内的数据和指令。本文中，我们提出了Cache Prune，这是一种防御方法，可以从输入提示上下文的KV缓存中识别和删除任务触发神经元。通过修剪此类神经元，我们鼓励LLM将输入提示上下文纯粹解释为数据，而不是指令遵循的线索。为了识别这些神经元，我们引入了一种由优先归因损失指导的神经归因机制，该机制只需少量样本即可实现有效归因，同时在修剪后保留反应质量。我们通过利用模型响应生成行为中固有的观察到的触发效应，进一步增强神经归因的功效。值得注意的是，我们的方法不会在提示上强加额外的格式，也不会引入额外的测试时LLM调用。实验表明，Cache Prune可以显着降低攻击成功率，同时保持干净的响应质量。



## **47. Memory Injection Attacks on LLM Agents via Query-Only Interaction**

通过仅查询交互对LLM代理进行内存注入攻击 cs.LG

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2503.03704v4) [paper-pdf](https://arxiv.org/pdf/2503.03704v4)

**Authors**: Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang

**Abstract**: Agents powered by large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, without assuming that the attacker can directly modify the memory bank of the agent. The attacker injects malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps corresponding to a different target query during the agent's execution of the victim user's query. Specifically, we introduce a sequence of bridging steps to link victim queries to the malicious reasoning steps. During the memory injection, we propose an indication prompt that guides the agent to autonomously generate similar bridging steps, with a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing later victim queries. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting the risk.

摘要: 由大型语言模型（LLM）支持的代理在各种复杂的现实世界应用程序中表现出了强大的能力。然而，当为演示而检索的过去记录是恶意的时，内存库受损的LLM代理可能很容易产生有害输出。在本文中，我们提出了一种新型的内存注入攻击MINJA，但没有假设攻击者可以直接修改代理的内存库。攻击者仅通过查询和输出观察与代理交互，将恶意记录注入内存库。这些恶意记录旨在在代理执行受害用户查询期间引发与不同目标查询相对应的一系列恶意推理步骤。具体来说，我们引入了一系列桥梁步骤，将受害者查询与恶意推理步骤联系起来。在内存注入过程中，我们提出了一个指示提示，引导代理自主生成类似的桥接步骤，逐步缩短策略，逐渐删除指示提示，这样的恶意记录将很容易被检索时，处理以后的受害者查询。我们对不同代理进行的广泛实验证明了MINJA在损害代理记忆方面的有效性。MINJA对执行的要求最低，使任何用户都能影响代理内存，从而凸显风险。



## **48. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

迈向智能和安全的云：大型语言模型增强主动防御 cs.CR

7 pages; Accepted by IEEE Communications Magazine

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2412.21051v4) [paper-pdf](https://arxiv.org/pdf/2412.21051v4)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided numerous benefits in our daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks such as Denial of Service (DoS). Recent advancements in the large language models (LLMs) offer promising solutions for security intelligence. By exploiting the powerful capabilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel defense architecture that proactively mitigates various DoS threats in cloud networks. LLM-PD can efficiently make decisions through comprehensive data analysis and sequential reasoning, as well as dynamically create and deploy actionable defense mechanisms. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. Our case study on three distinct DoS attacks demonstrates its remarkable ability in terms of defense effectiveness and efficiency when compared with other existing methods.

摘要: 云计算技术的快速发展和云应用程序数量的不断增加为我们的日常生活带来了诸多好处。然而，不同组件的多样性和复杂性对云安全构成了重大挑战，特别是在处理拒绝服务（Doc）等复杂且高级的网络攻击时。大型语言模型（LLM）的最新进展为安全情报提供了有前途的解决方案。通过利用语言理解、数据分析、任务推理、动作规划和代码生成方面的强大功能，我们提出了LLM-PD，这是一种新型防御架构，可以主动缓解云网络中的各种NOS威胁。LLM-PD可以通过全面的数据分析和顺序推理有效地做出决策，并动态创建和部署可操作的防御机制。此外，它可以根据从之前的交互中学到的经验灵活地自我进化，并在无需额外训练的情况下适应新的攻击场景。我们对三种不同的DPS攻击的案例研究表明，与其他现有方法相比，其在防御有效性和效率方面具有出色的能力。



## **49. A Fingerprint for Large Language Models**

大型语言模型的指纹 cs.CR

Updated by Hanzhou Wu, 8 pages

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2407.01235v2) [paper-pdf](https://arxiv.org/pdf/2407.01235v2)

**Authors**: Zhiguang Yang, Hanzhou Wu

**Abstract**: Recent advances confirm that large language models (LLMs) can achieve state-of-the-art performance across various tasks. However, due to the resource-intensive nature of training LLMs from scratch, it is urgent and crucial to protect the intellectual property of LLMs against infringement. This has motivated the authors in this paper to propose a novel black-box fingerprinting technique for LLMs. We firstly demonstrate that the outputs of LLMs span a unique vector space associated with each model. We model the problem of fingerprint authentication as the task of evaluating the similarity between the space of the victim model and the space of the suspect model. To tackle with this problem, we introduce two solutions: the first determines whether suspect outputs lie within the victim's subspace, enabling fast infringement detection; the second reconstructs a joint subspace to detect models modified via parameter-efficient fine-tuning (PEFT). Experiments indicate that the proposed method achieves superior performance in fingerprint verification and robustness against the PEFT attacks. This work reveals inherent characteristics of LLMs and provides a promising solution for protecting LLMs, ensuring efficiency, generality and practicality.

摘要: 最近的进展证实，大型语言模型（LLM）可以在各种任务中实现最先进的性能。然而，由于从头开始培训法学硕士的资源密集型性质，保护法学硕士的知识产权免遭侵权显得紧迫而至关重要。这促使本文作者为LLM提出了一种新型的黑匣子指纹技术。我们首先证明LLM的输出跨越与每个模型相关的唯一载体空间。我们将指纹认证问题建模为评估受害者模型空间和嫌疑人模型空间之间相似性的任务。为了解决这个问题，我们引入了两种解决方案：第一种解决方案确定可疑输出是否位于受害者的子空间内，从而实现快速侵权检测;第二种重建联合子空间以检测通过参数高效微调（PEFT）修改的模型。实验表明，该方法在指纹验证方面具有优越的性能和对PEFT攻击的鲁棒性。这项工作揭示了LLM的固有特征，并为保护LLM、确保效率、通用性和实用性提供了一个有前途的解决方案。



## **50. Gradient-Free Privacy Leakage in Federated Language Models through Selective Weight Tampering**

通过选择性权重篡改实现联邦语言模型中的无委托隐私泄露 cs.CR

21 pages (including bibliography and Appendix), Submitted to PETS'26

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2310.16152v4) [paper-pdf](https://arxiv.org/pdf/2310.16152v4)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) has become a key component in various language modeling applications such as machine translation, next-word prediction, and medical record analysis. These applications are trained on datasets from many FL participants that often include privacy-sensitive data, such as healthcare records, phone/credit card numbers, login credentials, etc. Although FL enables computation without necessitating clients to share their raw data, existing works show that privacy leakage is still probable in federated language models. In this paper, we present two novel findings on the leakage of privacy-sensitive user data from federated large language models without requiring access to gradients. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that a malicious FL participant can aggravate the leakage by tampering with the model's selective weights that are responsible for memorizing the sensitive training data of some other clients, even without any cooperation from the server. Our best-performing method increases the membership inference recall by 29% and achieves up to 71% private data reconstruction, evidently outperforming existing attacks that consider much stronger adversary capabilities. Lastly, we recommend a balanced suite of techniques for an FL client to defend against such privacy risk.

摘要: 联合学习（FL）已成为机器翻译、下一词预测和医疗记录分析等各种语言建模应用的关键组件。这些应用程序是在来自许多FL参与者的数据集上训练的，这些数据通常包括隐私敏感数据，例如医疗记录、电话/信用卡号码、登录凭证等。尽管FL可以在无需客户共享原始数据的情况下进行计算，但现有的工作表明，隐私泄露在联邦语言模型中仍然有可能。在本文中，我们提出了两项关于在不需要访问梯度的情况下从联邦大型语言模型泄露隐私敏感用户数据的新发现。首先，我们进行了一个关键观察，即FL中中间回合的模型快照可能会比最终训练的模型造成更大的隐私泄露。其次，我们发现恶意FL参与者可以通过篡改负责记忆其他一些客户端敏感训练数据的模型选择性权重来加剧泄漏，即使没有服务器的任何合作。我们性能最好的方法将成员推断召回率提高了29%，并实现了高达71%的私有数据重建，显然优于考虑对手能力更强的现有攻击。最后，我们为FL客户推荐一套平衡的技术来抵御此类隐私风险。



