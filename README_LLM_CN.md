# Latest Large Language Model Attack Papers
**update at 2025-05-28 14:52:07**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment**

通过特征最佳对齐对闭源MLLM的对抗攻击 cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21494v1) [paper-pdf](http://arxiv.org/pdf/2505.21494v1)

**Authors**: Xiaojun Jia, Sensen Gao, Simeng Qin, Tianyu Pang, Chao Du, Yihao Huang, Xinfeng Li, Yiming Li, Bo Li, Yang Liu

**Abstract**: Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples. While existing methods typically achieve targeted attacks by aligning global features-such as CLIP's [CLS] token-between adversarial and target samples, they often overlook the rich local information encoded in patch tokens. This leads to suboptimal alignment and limited transferability, particularly for closed-source models. To address this limitation, we propose a targeted transferable adversarial attack method based on feature optimal alignment, called FOA-Attack, to improve adversarial transfer capability. Specifically, at the global level, we introduce a global feature loss based on cosine similarity to align the coarse-grained features of adversarial samples with those of target samples. At the local level, given the rich local representations within Transformers, we leverage clustering techniques to extract compact local patterns to alleviate redundant local features. We then formulate local feature alignment between adversarial and target samples as an optimal transport (OT) problem and propose a local clustering optimal transport loss to refine fine-grained feature alignment. Additionally, we propose a dynamic ensemble model weighting strategy to adaptively balance the influence of multiple models during adversarial example generation, thereby further improving transferability. Extensive experiments across various models demonstrate the superiority of the proposed method, outperforming state-of-the-art methods, especially in transferring to closed-source MLLMs. The code is released at https://github.com/jiaxiaojunQAQ/FOA-Attack.

摘要: 多模式大型语言模型（MLLM）仍然容易受到可转移的对抗示例的影响。虽然现有方法通常通过在对抗样本和目标样本之间对齐全局特征（例如CLIP的[LIS]标记）来实现有针对性的攻击，但它们经常忽视补丁令牌中编码的丰富本地信息。这导致次优的对齐和有限的可移植性，特别是对于闭源模型。为了解决这一局限性，我们提出了一种基于特征最优对齐的有针对性的可转移对抗攻击方法，称为FOA-Attack，以提高对抗转移能力。具体来说，在全球层面，我们引入了基于cos相似性的全球特征损失，以将对抗样本的粗粒度特征与目标样本的粗粒度特征对齐。在局部层面，鉴于变形金刚中丰富的局部表示，我们利用集群技术来提取紧凑的局部模式，以减轻冗余的局部特征。然后，我们将对抗样本和目标样本之间的局部特征对齐公式化为最优传输（OT）问题，并提出局部集群最优传输损失来细化细粒度特征对齐。此外，我们还提出了一种动态集成模型加权策略，以自适应地平衡对抗性示例生成过程中多个模型的影响，从而进一步提高可移植性。跨各种模型的广泛实验证明了所提出方法的优越性，优于最先进的方法，特别是在转移到闭源MLLM方面。该代码发布于https://github.com/jiaxiaojunQAQ/FOA-Attack。



## **2. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

GUARD：神经代码生成中基于双智能体的思维链后门防御 cs.SE

under review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21425v1) [paper-pdf](http://arxiv.org/pdf/2505.21425v1)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.

摘要: 随着大型语言模型在代码生成中的广泛应用，最近的研究表明，采用额外的思想链生成模型可以通过提供显式推理步骤来显着提高代码生成性能。然而，作为外部组件，CoT模型特别容易受到后门攻击，而现有的防御机制往往无法有效检测到后门攻击。为了应对这一挑战，我们提出了GUARD，这是一种新型双代理防御框架，专门设计用于对抗神经代码生成中的CoT后门攻击。GUARD集成了两个核心组件：GUARD-Judge，通过全面分析识别可疑的CoT步骤和潜在触发因素，以及GUARD-Repair，采用检索增强生成方法来为识别的异常重新生成安全CoT步骤。实验结果表明，GUARD有效地缓解了攻击，同时保持生成质量，推进了安全代码生成系统。



## **3. Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space**

打破天花板：通过扩大战略空间探索越狱袭击的潜力 cs.CR

19 pages, 20 figures, accepted by ACL 2025, Findings

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21277v1) [paper-pdf](http://arxiv.org/pdf/2505.21277v1)

**Authors**: Yao Huang, Yitong Sun, Shouwei Ruan, Yichi Zhang, Yinpeng Dong, Xingxing Wei

**Abstract**: Large Language Models (LLMs), despite advanced general capabilities, still suffer from numerous safety risks, especially jailbreak attacks that bypass safety protocols. Understanding these vulnerabilities through black-box jailbreak attacks, which better reflect real-world scenarios, offers critical insights into model robustness. While existing methods have shown improvements through various prompt engineering techniques, their success remains limited against safety-aligned models, overlooking a more fundamental problem: the effectiveness is inherently bounded by the predefined strategy spaces. However, expanding this space presents significant challenges in both systematically capturing essential attack patterns and efficiently navigating the increased complexity. To better explore the potential of expanding the strategy space, we address these challenges through a novel framework that decomposes jailbreak strategies into essential components based on the Elaboration Likelihood Model (ELM) theory and develops genetic-based optimization with intention evaluation mechanisms. To be striking, our experiments reveal unprecedented jailbreak capabilities by expanding the strategy space: we achieve over 90% success rate on Claude-3.5 where prior methods completely fail, while demonstrating strong cross-model transferability and surpassing specialized safeguard models in evaluation accuracy. The code is open-sourced at: https://github.com/Aries-iai/CL-GSO.

摘要: 尽管大型语言模型（LLM）具有先进的通用功能，但仍然面临许多安全风险，尤其是绕过安全协议的越狱攻击。通过黑匣子越狱攻击来了解这些漏洞（更好地反映了现实世界的场景），可以为模型稳健性提供重要见解。虽然现有方法通过各种即时工程技术表现出了改进，但它们的成功仍然局限于安全性一致的模型，忽视了一个更根本的问题：有效性本质上受到预定义的策略空间的限制。然而，扩展这一空间在系统性捕获基本攻击模式和有效应对日益增加的复杂性方面带来了重大挑战。为了更好地探索扩大策略空间的潜力，我们通过一个新颖的框架来应对这些挑战，该框架基于埃斯珀似然模型（ELM）理论将越狱策略分解为基本组件，并开发具有意图评估机制的基于遗传的优化。引人注目的是，我们的实验通过扩大策略空间揭示了前所未有的越狱能力：在现有方法完全失败的情况下，我们在Claude-3.5上实现了90%以上的成功率，同时展示了强大的跨模型可移植性，并在评估准确性方面超越了专业保障模型。该代码的开源网址：https://github.com/Aries-iai/CL-GSO。



## **4. JavaSith: A Client-Side Framework for Analyzing Potentially Malicious Extensions in Browsers, VS Code, and NPM Packages**

JavSith：一个用于分析浏览器、VS代码和NPM包中潜在恶意扩展的客户端框架 cs.CR

28 pages , 11 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21263v1) [paper-pdf](http://arxiv.org/pdf/2505.21263v1)

**Authors**: Avihay Cohen

**Abstract**: Modern software supply chains face an increasing threat from malicious code hidden in trusted components such as browser extensions, IDE extensions, and open-source packages. This paper introduces JavaSith, a novel client-side framework for analyzing potentially malicious extensions in web browsers, Visual Studio Code (VSCode), and Node's NPM packages. JavaSith combines a runtime sandbox that emulates browser/Node.js extension APIs (with a ``time machine'' to accelerate time-based triggers) with static analysis and a local large language model (LLM) to assess risk from code and metadata. We present the design and architecture of JavaSith, including techniques for intercepting extension behavior over simulated time and extracting suspicious patterns. Through case studies on real-world attacks (such as a supply-chain compromise of a Chrome extension and malicious VSCode extensions installing cryptominers), we demonstrate how JavaSith can catch stealthy malicious behaviors that evade traditional detection. We evaluate the framework's effectiveness and discuss its limitations and future enhancements. JavaSith's client-side approach empowers end-users/organizations to vet extensions and packages before trustingly integrating them into their environments.

摘要: 现代软件供应链面临着隐藏在浏览器扩展、IDE扩展和开源包等受信任组件中的恶意代码越来越大的威胁。本文介绍了Java Sith，这是一种新型客户端框架，用于分析Web浏览器中潜在的恶意扩展、Visual Studio Code（VSCode）和NPM包。JavSith将模拟浏览器/Node. js扩展API（带有“时间机”来加速基于时间的触发器）的运行时沙箱与静态分析和本地大型语言模型（LLM）相结合，以评估代码和元数据的风险。我们介绍了Java Sith的设计和体系结构，包括在模拟时间内拦截扩展行为和提取可疑模式的技术。通过对真实世界攻击的案例研究（例如Chrome扩展的供应链妥协和安装cryptominers的恶意VSCode扩展），我们展示了JavaSith如何捕获逃避传统检测的隐形恶意行为。我们评估框架的有效性，并讨论其局限性和未来的增强。JavaSith的客户端方法使最终用户/组织能够在将扩展和包可靠地集成到其环境中之前对其进行审查。



## **5. SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA**

SHE-LoRA：用于使用异类LoRA进行联邦调优的选择性homomorm加密 cs.CR

24 pages, 13 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21051v1) [paper-pdf](http://arxiv.org/pdf/2505.21051v1)

**Authors**: Jianmin Liu, Li Yan, Borui Li, Lei Yu, Chao Shen

**Abstract**: Federated fine-tuning of large language models (LLMs) is critical for improving their performance in handling domain-specific tasks. However, prior work has shown that clients' private data can actually be recovered via gradient inversion attacks. Existing privacy preservation techniques against such attacks typically entail performance degradation and high costs, making them ill-suited for clients with heterogeneous data distributions and device capabilities. In this paper, we propose SHE-LoRA, which integrates selective homomorphic encryption (HE) and low-rank adaptation (LoRA) to enable efficient and privacy-preserving federated tuning of LLMs in cross-device environment. Heterogeneous clients adaptively select partial model parameters for homomorphic encryption based on parameter sensitivity assessment, with the encryption subset obtained via negotiation. To ensure accurate model aggregation, we design a column-aware secure aggregation method and customized reparameterization techniques to align the aggregation results with the heterogeneous device capabilities of clients. Extensive experiments demonstrate that SHE-LoRA maintains performance comparable to non-private baselines, achieves strong resistance to the state-of-the-art attacks, and significantly reduces communication overhead by 94.901\% and encryption computation overhead by 99.829\%, compared to baseline. Our code is accessible at https://anonymous.4open.science/r/SHE-LoRA-8D84.

摘要: 大型语言模型（LLM）的联合微调对于提高其处理特定领域任务的性能至关重要。然而，之前的工作表明，客户的私人数据实际上可以通过梯度倒置攻击恢复。针对此类攻击的现有隐私保护技术通常会导致性能下降和成本高，使其不适合具有异类数据分布和设备功能的客户端。本文中，我们提出了SHE-LoRA，它集成了选择性homomorphic加密（HE）和低等级自适应（LoRA），以实现跨设备环境中LLM的高效且保护隐私的联邦调优。异类客户端根据参数敏感度评估自适应地选择部分模型参数进行同质加密，加密子集通过协商获得。为了确保准确的模型聚合，我们设计了一种列感知的安全聚合方法和自定义的重新参数化技术，以使聚合结果与客户端的异类设备能力保持一致。大量实验表明，SHE-LoRA保持了与非私有基线相当的性能，对最先进的攻击具有强大的抵抗力，并与基线相比，将通信负担显着减少了94.901%，加密计算负担显着减少了99.829%。我们的代码可在https://anonymous.4open.science/r/SHE-LoRA-8D84上访问。



## **6. BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models**

BitHydra：针对大型语言模型的位翻转推理成本攻击 cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.16670v2) [paper-pdf](http://arxiv.org/pdf/2505.16670v2)

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs.

摘要: 大型语言模型（LLM）在广泛的应用程序中表现出令人印象深刻的能力，但其不断增加的规模和资源需求使它们容易受到推理成本攻击，攻击者诱导受害者LLM生成尽可能长的输出内容。在本文中，我们回顾了现有的推理成本攻击，并揭示了这些方法很难产生大规模的恶意影响，因为它们是自瞄准的，攻击者也是用户，因此必须仅通过输入来执行攻击，其生成的内容将由LLM收费并且只能直接影响自己。受这些发现的启发，本文引入了一种新型的推理成本攻击（称为“位翻转推理成本攻击”），其目标是受害者模型本身，而不是其输入。具体来说，我们设计了一种简单而有效的方法（称为“BitHydra”）来有效地翻转模型参数的关键部分。该过程由损失函数指导，该函数旨在<EOS>通过高效的关键位搜索算法抑制令牌的概率，从而明确定义攻击目标并实现有效的优化。我们在int8和float 16设置下对11个LLM（参数范围从1.5B到14B）上评估了我们的方法。实验结果表明，只需4个搜索样本和少至3位翻转，BitHydra就可以强制100%的测试提示达到最大生成长度（例如，2048个令牌），突出了其效率，可扩展性和跨看不见的输入的强大可转移性。



## **7. IRCopilot: Automated Incident Response with Large Language Models**

IRCopilot：使用大型语言模型的自动化事件响应 cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20945v1) [paper-pdf](http://arxiv.org/pdf/2505.20945v1)

**Authors**: Xihuan Lin, Jie Zhang, Gelei Deng, Tianzhe Liu, Xiaolong Liu, Changcai Yang, Tianwei Zhang, Qing Guo, Riqing Chen

**Abstract**: Incident response plays a pivotal role in mitigating the impact of cyber attacks. In recent years, the intensity and complexity of global cyber threats have grown significantly, making it increasingly challenging for traditional threat detection and incident response methods to operate effectively in complex network environments. While Large Language Models (LLMs) have shown great potential in early threat detection, their capabilities remain limited when it comes to automated incident response after an intrusion. To address this gap, we construct an incremental benchmark based on real-world incident response tasks to thoroughly evaluate the performance of LLMs in this domain. Our analysis reveals several key challenges that hinder the practical application of contemporary LLMs, including context loss, hallucinations, privacy protection concerns, and their limited ability to provide accurate, context-specific recommendations. In response to these challenges, we propose IRCopilot, a novel framework for automated incident response powered by LLMs. IRCopilot mimics the three dynamic phases of a real-world incident response team using four collaborative LLM-based session components. These components are designed with clear divisions of responsibility, reducing issues such as hallucinations and context loss. Our method leverages diverse prompt designs and strategic responsibility segmentation, significantly improving the system's practicality and efficiency. Experimental results demonstrate that IRCopilot outperforms baseline LLMs across key benchmarks, achieving sub-task completion rates of 150%, 138%, 136%, 119%, and 114% for various response tasks. Moreover, IRCopilot exhibits robust performance on public incident response platforms and in real-world attack scenarios, showcasing its strong applicability.

摘要: 事件响应在减轻网络攻击的影响方面发挥着关键作用。近年来，全球网络威胁的强度和复杂性显着增长，使得传统的威胁检测和事件响应方法在复杂网络环境中有效运作面临越来越大的挑战。虽然大型语言模型（LLM）在早期威胁检测方面表现出了巨大的潜力，但在入侵后自动化事件响应方面，它们的能力仍然有限。为了解决这一差距，我们基于现实世界的事件响应任务构建了一个增量基准，以彻底评估LLM在该领域的性能。我们的分析揭示了阻碍当代LLM实际应用的几个关键挑战，包括上下文丢失、幻觉、隐私保护问题，以及它们提供准确的、针对特定上下文的建议的能力有限。为了应对这些挑战，我们提出了IRCopilot，这是一个由LLM支持的自动化事件响应的新型框架。IRCopilot使用四个基于LLM的协作会话组件模拟现实世界事件响应团队的三个动态阶段。这些组件的设计有明确的责任分工，减少了幻觉和上下文丢失等问题。我们的方法利用多样化的提示设计和战略责任细分，显着提高了系统的实用性和效率。实验结果表明，IRCopilot在关键基准上的表现优于基线LLM，各种响应任务的子任务完成率分别为150%、138%、136%、119%和114%。此外，IRCopilot在公共事件响应平台和现实世界的攻击场景中表现出稳健的性能，展示了其强大的适用性。



## **8. Concealment of Intent: A Game-Theoretic Analysis**

意图的隐瞒：游戏理论分析 cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20841v1) [paper-pdf](http://arxiv.org/pdf/2505.20841v1)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.

摘要: 随着大型语言模型（LLM）的能力越来越强，对其安全部署的担忧也越来越大。尽管已经引入了对齐机制来阻止滥用，但它们仍然容易受到精心设计的对抗提示的影响。在这项工作中，我们提出了一种可扩展的攻击策略：意图隐藏对抗提示，通过技能的组合来隐藏恶意意图。我们开发了一个博弈论框架来建模此类攻击与应用提示和响应过滤的防御系统之间的相互作用。我们的分析确定了平衡点并揭示了攻击者的结构优势。为了应对这些威胁，我们提出并分析了一种针对意图隐藏攻击的防御机制。从经验上讲，我们验证了攻击对一系列恶意行为的多个现实世界LLM的有效性，展示了比现有对抗提示技术的明显优势。



## **9. MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems**

MedSentry：了解和缓解医学LLM多主体系统中的安全风险 cs.MA

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20824v1) [paper-pdf](http://arxiv.org/pdf/2505.20824v1)

**Authors**: Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, Yang Gao

**Abstract**: As large language models (LLMs) are increasingly deployed in healthcare, ensuring their safety, particularly within collaborative multi-agent configurations, is paramount. In this paper we introduce MedSentry, a benchmark comprising 5 000 adversarial medical prompts spanning 25 threat categories with 100 subthemes. Coupled with this dataset, we develop an end-to-end attack-defense evaluation pipeline to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) withstand attacks from 'dark-personality' agents. Our findings reveal critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For instance, SharedPool's open information sharing makes it highly susceptible, whereas Decentralized architectures exhibit greater resilience thanks to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring system safety to near-baseline levels. MedSentry thus furnishes both a rigorous evaluation framework and practical defense strategies that guide the design of safer LLM-based multi-agent systems in medical domains.

摘要: 随着大型语言模型（LLM）越来越多地部署在医疗保健中，确保其安全性，特别是在协作多代理配置中，至关重要。在本文中，我们介绍了MedSentry，这是一个基准，由5000个对抗性医疗提示组成，涵盖25个威胁类别和100个子主题。与此数据集相结合，我们开发了一个端到端的攻击防御评估管道，以系统地分析四种代表性的多智能体布局（Layers、SharedPool、Centralized和Decentralized）如何抵御来自“黑暗人格”智能体的攻击。我们的研究结果揭示了这些架构如何处理信息污染和维持稳健决策的关键差异，暴露了其潜在的脆弱性机制。例如，SharedPool的开放信息共享使其高度容易受到影响，而去中心化架构由于固有的冗余和隔离而表现出更大的弹性。为了减轻这些风险，我们提出了一种个性规模的检测和纠正机制，该机制可以识别和恢复恶意代理，将系统安全性恢复到接近基线的水平。因此，MedSentry提供了严格的评估框架和实用的防御策略，指导医学领域更安全的基于LLM的多智能体系统的设计。



## **10. TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent**

TrojanStego：你的语言模型可以秘密地成为隐写隐私泄露代理 cs.CL

9 pages, 5 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20118v2) [paper-pdf](http://arxiv.org/pdf/2505.20118v2)

**Authors**: Dominik Meier, Jan Philip Wahle, Paul Röttger, Terry Ruas, Bela Gipp

**Abstract**: As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.

摘要: 随着大型语言模型（LLM）集成到敏感工作流程中，人们越来越担心它们泄露机密信息的可能性。我们提出了TrojanStego，这是一种新型威胁模型，其中对手微调LLM，通过语言隐写术将敏感的上下文信息嵌入到看起来自然的输出中，而不需要对推理输入进行显式控制。我们引入了一个分类法，概述了受影响的LLM的风险因素，并使用它来评估威胁的风险状况。为了实现TrojanStego，我们提出了一种基于词汇划分的实用编码方案，LLM可以通过微调学习。实验结果表明，受攻击的模型在发出的提示上以87%的准确率可靠地传输32位秘密，使用三代多数投票的准确率达到97%以上。此外，它们保持高实用性，可以逃避人类检测，并保持一致性。这些结果凸显了一类新型LLM数据泄露攻击，这些攻击是被动的、隐蔽的、实用的且危险的。



## **11. Improved Representation Steering for Language Models**

改进的语言模型引导表示 cs.CL

46 pages, 23 figures, preprint

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20809v1) [paper-pdf](http://arxiv.org/pdf/2505.20809v1)

**Authors**: Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, Christopher Potts

**Abstract**: Steering methods for language models (LMs) seek to provide fine-grained and interpretable control over model generations by variously changing model inputs, weights, or representations to adjust behavior. Recent work has shown that adjusting weights or representations is often less effective than steering by prompting, for instance when wanting to introduce or suppress a particular concept. We demonstrate how to improve representation steering via our new Reference-free Preference Steering (RePS), a bidirectional preference-optimization objective that jointly does concept steering and suppression. We train three parameterizations of RePS and evaluate them on AxBench, a large-scale model steering benchmark. On Gemma models with sizes ranging from 2B to 27B, RePS outperforms all existing steering methods trained with a language modeling objective and substantially narrows the gap with prompting -- while promoting interpretability and minimizing parameter count. In suppression, RePS matches the language-modeling objective on Gemma-2 and outperforms it on the larger Gemma-3 variants while remaining resilient to prompt-based jailbreaking attacks that defeat prompting. Overall, our results suggest that RePS provides an interpretable and robust alternative to prompting for both steering and suppression.

摘要: 语言模型（LM）的引导方法试图通过各种改变模型输入、权重或表示来调整行为来提供对模型生成的细粒度且可解释的控制。最近的工作表明，调整权重或表示通常不如通过提示引导有效，例如当想要引入或抑制特定概念时。我们演示了如何通过新的无引用偏好引导（RePS）来改进表示引导，RePS是一个双向偏好优化目标，可以联合进行概念引导和抑制。我们训练RePS的三个参数化，并在AxBench（一个大型模型转向基准）上对其进行评估。在大小从2B到27 B的Gemma模型上，RePS优于所有现有的使用语言建模目标训练的转向方法，并大大缩小了与提示的差距-同时提高了可解释性并最大限度地减少了参数数量。在抑制方面，RePS匹配Gemma-2上的语言建模目标，并在更大的Gemma-3变体上优于它，同时保持对失败提示的基于密码的越狱攻击的弹性。总的来说，我们的研究结果表明，RePS提供了一个可解释的和强大的替代，以促进转向和抑制。



## **12. Forewarned is Forearmed: A Survey on Large Language Model-based Agents in Autonomous Cyberattacks**

预先警告就是预先武装：自主网络攻击中基于大型语言模型的代理的调查 cs.NI

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.12786v2) [paper-pdf](http://arxiv.org/pdf/2505.12786v2)

**Authors**: Minrui Xu, Jiani Fan, Xinyu Huang, Conghao Zhou, Jiawen Kang, Dusit Niyato, Shiwen Mao, Zhu Han, Xuemin, Shen, Kwok-Yan Lam

**Abstract**: With the continuous evolution of Large Language Models (LLMs), LLM-based agents have advanced beyond passive chatbots to become autonomous cyber entities capable of performing complex tasks, including web browsing, malicious code and deceptive content generation, and decision-making. By significantly reducing the time, expertise, and resources, AI-assisted cyberattacks orchestrated by LLM-based agents have led to a phenomenon termed Cyber Threat Inflation, characterized by a significant reduction in attack costs and a tremendous increase in attack scale. To provide actionable defensive insights, in this survey, we focus on the potential cyber threats posed by LLM-based agents across diverse network systems. Firstly, we present the capabilities of LLM-based cyberattack agents, which include executing autonomous attack strategies, comprising scouting, memory, reasoning, and action, and facilitating collaborative operations with other agents or human operators. Building on these capabilities, we examine common cyberattacks initiated by LLM-based agents and compare their effectiveness across different types of networks, including static, mobile, and infrastructure-free paradigms. Moreover, we analyze threat bottlenecks of LLM-based agents across different network infrastructures and review their defense methods. Due to operational imbalances, existing defense methods are inadequate against autonomous cyberattacks. Finally, we outline future research directions and potential defensive strategies for legacy network systems.

摘要: 随着大型语言模型（LLM）的不断发展，基于LLM的代理已经超越被动聊天机器人，成为能够执行复杂任务的自治网络实体，包括网络浏览、恶意代码和欺骗性内容生成以及决策。通过显着减少时间、专业知识和资源，由LLM代理策划的人工智能辅助网络攻击导致了一种称为网络威胁通货膨胀的现象，其特征是攻击成本显着降低和攻击规模显着增加。为了提供可操作的防御见解，在本调查中，我们重点关注基于LLM的代理在不同网络系统中构成的潜在网络威胁。首先，我们介绍了基于LLM的网络攻击代理的能力，其中包括执行自主攻击策略，包括侦察、记忆、推理和行动，以及促进与其他代理或人类操作员的协作操作。基于这些功能，我们研究了基于LLM的代理发起的常见网络攻击，并比较了它们在不同类型网络（包括静态，移动和无基础设施模式）中的有效性。此外，我们分析了基于LLM的代理在不同的网络基础设施的威胁瓶颈，并审查其防御方法。由于操作不平衡，现有的防御方法不足以应对自主网络攻击。最后，我们概述了未来的研究方向和潜在的防御策略的遗留网络系统。



## **13. $C^3$-Bench: The Things Real Disturbing LLM based Agent in Multi-Tasking**

$C ' 3 $-Bench：多任务中令人不安的LLM代理人真正不安的事情 cs.AI

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.18746v2) [paper-pdf](http://arxiv.org/pdf/2505.18746v2)

**Authors**: Peijie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, Feng Zhang

**Abstract**: Agents based on large language models leverage tools to modify environments, revolutionizing how AI interacts with the physical world. Unlike traditional NLP tasks that rely solely on historical dialogue for responses, these agents must consider more complex factors, such as inter-tool relationships, environmental feedback and previous decisions, when making choices. Current research typically evaluates agents via multi-turn dialogues. However, it overlooks the influence of these critical factors on agent behavior. To bridge this gap, we present an open-source and high-quality benchmark $C^3$-Bench. This benchmark integrates attack concepts and applies univariate analysis to pinpoint key elements affecting agent robustness. In concrete, we design three challenges: navigate complex tool relationships, handle critical hidden information and manage dynamic decision paths. Complementing these challenges, we introduce fine-grained metrics, innovative data collection algorithms and reproducible evaluation methods. Extensive experiments are conducted on 49 mainstream agents, encompassing general fast-thinking, slow-thinking and domain-specific models. We observe that agents have significant shortcomings in handling tool dependencies, long context information dependencies and frequent policy-type switching. In essence, $C^3$-Bench aims to expose model vulnerabilities through these challenges and drive research into the interpretability of agent performance. The benchmark is publicly available at https://github.com/yupeijei1997/C3-Bench.

摘要: 基于大型语言模型的代理利用工具来修改环境，彻底改变了人工智能与物理世界交互的方式。与仅依赖历史对话来做出反应的传统NLP任务不同，这些代理人在做出选择时必须考虑更复杂的因素，例如工具间关系、环境反馈和之前的决策。当前的研究通常通过多轮对话来评估代理人。然而，它忽视了这些关键因素对代理行为的影响。为了弥合这一差距，我们提出了一个开源且高质量的基准$C#3 $-Bench。该基准测试集成了攻击概念并应用单变量分析来确定影响代理稳健性的关键元素。具体而言，我们设计了三个挑战：导航复杂的工具关系、处理关键的隐藏信息以及管理动态决策路径。为了补充这些挑战，我们引入了细粒度指标、创新的数据收集算法和可重复的评估方法。广泛的实验进行了49个主流代理，包括一般的快思维，慢思维和特定领域的模型。我们观察到，代理有显着的缺点，在处理工具的依赖关系，长上下文信息的依赖关系和频繁的政策类型切换。本质上，$C^3$-Bench旨在通过这些挑战暴露模型漏洞，并推动对代理性能可解释性的研究。该基准可在https://github.com/yupeijei1997/C3-Bench上公开获得。



## **14. Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models**

超越效率：揭露小型语言模型中越狱攻击的潜在威胁 cs.CR

Accepted to ACL 2025 findings

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2502.19883v3) [paper-pdf](http://arxiv.org/pdf/2502.19883v3)

**Authors**: Sibo Yi, Tianshuo Cong, Xinlei He, Qi Li, Jiaxing Song

**Abstract**: Small language models (SLMs) have become increasingly prominent in the deployment on edge devices due to their high efficiency and low computational cost. While researchers continue to advance the capabilities of SLMs through innovative training strategies and model compression techniques, the security risks of SLMs have received considerably less attention compared to large language models (LLMs).To fill this gap, we provide a comprehensive empirical study to evaluate the security performance of 13 state-of-the-art SLMs under various jailbreak attacks. Our experiments demonstrate that most SLMs are quite susceptible to existing jailbreak attacks, while some of them are even vulnerable to direct harmful prompts.To address the safety concerns, we evaluate several representative defense methods and demonstrate their effectiveness in enhancing the security of SLMs. We further analyze the potential security degradation caused by different SLM techniques including architecture compression, quantization, knowledge distillation, and so on. We expect that our research can highlight the security challenges of SLMs and provide valuable insights to future work in developing more robust and secure SLMs.

摘要: 小型语言模型（SLC）因其高效率和低计算成本而在边缘设备上的部署中变得越来越重要。虽然研究人员不断通过创新的训练策略和模型压缩技术来提高CRM的能力，但与大型语言模型（LLM）相比，CRM的安全风险受到的关注要少得多。为了填补这一空白，我们提供了一项全面的实证研究来评估13种最先进的CRM在各种越狱攻击下的安全性能。我们的实验表明，大多数Slms都很容易受到现有的越狱攻击，而其中一些甚至很容易受到直接的有害提示。为了解决安全问题，我们评估了几种代表性的防御方法，并展示了它们在增强Slms安全性方面的有效性。我们进一步分析了架构压缩、量化、知识提炼等不同的SLA技术所导致的潜在安全降级。我们希望我们的研究能够突出SLC的安全挑战，并为未来开发更强大、更安全的SLC的工作提供有价值的见解。



## **15. Capability-Based Scaling Laws for LLM Red-Teaming**

LLM红色团队基于能力的缩放法则 cs.AI

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20162v1) [paper-pdf](http://arxiv.org/pdf/2505.20162v1)

**Authors**: Alexander Panfilov, Paul Kassianik, Maksym Andriushchenko, Jonas Geiping

**Abstract**: As large language models grow in capability and agency, identifying vulnerabilities through red-teaming becomes vital for safe deployment. However, traditional prompt-engineering approaches may prove ineffective once red-teaming turns into a weak-to-strong problem, where target models surpass red-teamers in capabilities. To study this shift, we frame red-teaming through the lens of the capability gap between attacker and target. We evaluate more than 500 attacker-target pairs using LLM-based jailbreak attacks that mimic human red-teamers across diverse families, sizes, and capability levels. Three strong trends emerge: (i) more capable models are better attackers, (ii) attack success drops sharply once the target's capability exceeds the attacker's, and (iii) attack success rates correlate with high performance on social science splits of the MMLU-Pro benchmark. From these trends, we derive a jailbreaking scaling law that predicts attack success for a fixed target based on attacker-target capability gap. These findings suggest that fixed-capability attackers (e.g., humans) may become ineffective against future models, increasingly capable open-source models amplify risks for existing systems, and model providers must accurately measure and control models' persuasive and manipulative abilities to limit their effectiveness as attackers.

摘要: 随着大型语言模型能力和代理能力的增长，通过红色团队识别漏洞对于安全部署变得至关重要。然而，一旦红色团队变成一个从弱到强的问题，即目标模型的能力超过红色团队，传统的预算工程方法可能会被证明无效。为了研究这种转变，我们通过攻击者和目标之间的能力差距来构建红色团队。我们使用基于LLM的越狱攻击来评估500多个攻击者目标对，这些攻击者目标对模拟不同家庭、规模和能力水平的人类红队成员。出现了三个强有力的趋势：（i）更有能力的模型是更好的攻击者，（ii）一旦目标的能力超过攻击者的能力，攻击成功率就会急剧下降，（iii）攻击成功率与MMLU-Pro基准的社会科学分裂的高性能相关。从这些趋势中，我们得出了一个越狱的比例法则，预测攻击成功的基础上攻击目标的能力差距的固定目标。这些发现表明，固定能力的攻击者（例如，人类）可能会对未来的模型变得无效，越来越强大的开源模型放大了现有系统的风险，模型提供商必须准确地测量和控制模型的说服和操纵能力，以限制其作为攻击者的有效性。



## **16. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

PandaGuard：针对越狱攻击的LLM安全性系统评估 cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.13862v3) [paper-pdf](http://arxiv.org/pdf/2505.13862v3)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.

摘要: 大型语言模型（LLM）已经取得了卓越的能力，但仍然容易受到被称为越狱的对抗性提示的影响，这可能会绕过安全对齐并引发有害的输出。尽管LLM安全研究的努力越来越多，但现有的评估往往是分散的，集中在孤立的攻击或防御技术上，缺乏系统的，可重复的分析。在这项工作中，我们引入了PandaGuard，一个统一的模块化框架，将LLM越狱安全建模为一个由攻击者，防御者和法官组成的多代理系统。我们的框架实现了19种攻击方法和12种防御机制，以及多种判断策略，所有这些都在一个灵活的插件架构中，支持多种LLM接口，多种交互模式和配置驱动的实验，从而增强了可重复性和实际部署。基于这个框架，我们开发了PandaBench，这是一个全面的基准，可评估49个LLM和各种判断方法之间的相互作用，需要超过30亿个代币来执行。我们的广泛评估揭示了对模型漏洞、国防成本-性能权衡和判断一致性的关键见解。我们发现，没有一种防御在所有维度上都是最佳的，而且判断分歧会在安全评估中引入非平凡的方差。我们发布代码、配置和评估结果，以支持LLM安全性方面的透明和可重复研究。



## **17. Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under Black-box Settings**

螃蟹：黑匣子设置下通过自动生成来消耗资源进行LLM-NOS攻击 cs.CL

22 pages, 8 figures, 11 tables

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2412.13879v4) [paper-pdf](http://arxiv.org/pdf/2412.13879v4)

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks yet still are vulnerable to external threats, particularly LLM Denial-of-Service (LLM-DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, existing studies predominantly focus on white-box attacks, leaving black-box scenarios underexplored. In this paper, we introduce Auto-Generation for LLM-DoS (AutoDoS) attack, an automated algorithm designed for black-box LLMs. AutoDoS constructs the DoS Attack Tree and expands the node coverage to achieve effectiveness under black-box conditions. By transferability-driven iterative optimization, AutoDoS could work across different models in one prompt. Furthermore, we reveal that embedding the Length Trojan allows AutoDoS to bypass existing defenses more effectively. Experimental results show that AutoDoS significantly amplifies service response latency by over 250$\times\uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our work provides a new perspective on LLM-DoS attacks and security defenses. Our code is available at https://github.com/shuita2333/AutoDoS.

摘要: 大型语言模型（LLM）在不同任务中表现出了出色的性能，但仍然容易受到外部威胁，尤其是LLM拒绝服务（LLM-NOS）攻击。具体来说，LLM-NOS攻击旨在耗尽计算资源并阻止服务。然而，现有的研究主要集中在白盒攻击上，对黑匣子场景的研究不足。本文中，我们介绍了LLM-DPS攻击的自动生成（AutoDock），这是一种为黑匣子LLM设计的自动算法。AutoDock构建了DPS攻击树，并扩大了节点覆盖范围，以在黑匣子条件下实现有效性。通过可移植性驱动的迭代优化，AutoDock可以在一次提示内跨不同的模型工作。此外，我们还发现，嵌入长度特洛伊木马可以让AutoDock更有效地绕过现有的防御。实验结果表明，AutoDock将服务响应延迟显着放大了超过250 $\times\uparrow $，导致在图形处理器利用率和内存使用率方面严重消耗资源。我们的工作为LLM-NOS攻击和安全防御提供了新的视角。我们的代码可在https://github.com/shuita2333/AutoDoS上获取。



## **18. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.05050v3) [paper-pdf](http://arxiv.org/pdf/2504.05050v3)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **19. Attention! You Vision Language Model Could Be Maliciously Manipulated**

注意！您的视觉语言模型可能被恶意操纵 cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19911v1) [paper-pdf](http://arxiv.org/pdf/2505.19911v1)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets.

摘要: 大型视觉语言模型（VLM）在理解复杂的现实世界场景和支持数据驱动的决策流程方面取得了显着的成功。然而，VLM对对抗性示例（无论是文本还是图像）表现出显着的脆弱性，这可能会导致各种对抗性结果，例如越狱、劫持和幻觉等。在这项工作中，我们从经验和理论上证明了VLM特别容易受到基于图像的对抗示例的影响，其中不可感知的扰动可以精确地操纵每个输出令牌。为此，我们提出了一种名为视觉语言模型操纵攻击（VMA）的新型攻击，该攻击将一阶和二阶动量优化技术与可微转换机制集成在一起，以有效地优化对抗性扰动。值得注意的是，VMA可以是一把双刃剑：它可以被用来实施各种攻击，例如越狱、劫持、隐私泄露、拒绝服务和海绵示例的生成等，同时允许注入水印以进行版权保护。广泛的实证评估证实了VMA在不同场景和数据集中的有效性和普遍性。



## **20. CPA-RAG:Covert Poisoning Attacks on Retrieval-Augmented Generation in Large Language Models**

CPA-RAG：对大型语言模型中检索增强生成的隐蔽中毒攻击 cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19864v1) [paper-pdf](http://arxiv.org/pdf/2505.19864v1)

**Authors**: Chunyang Li, Junwei Zhang, Anda Cheng, Zhuo Ma, Xinghua Li, Jianfeng Ma

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but its openness introduces vulnerabilities that can be exploited by poisoning attacks. Existing poisoning methods for RAG systems have limitations, such as poor generalization and lack of fluency in adversarial texts. In this paper, we propose CPA-RAG, a black-box adversarial framework that generates query-relevant texts capable of manipulating the retrieval process to induce target answers. The proposed method integrates prompt-based text generation, cross-guided optimization through multiple LLMs, and retriever-based scoring to construct high-quality adversarial samples. We conduct extensive experiments across multiple datasets and LLMs to evaluate its effectiveness. Results show that the framework achieves over 90\% attack success when the top-k retrieval setting is 5, matching white-box performance, and maintains a consistent advantage of approximately 5 percentage points across different top-k values. It also outperforms existing black-box baselines by 14.5 percentage points under various defense strategies. Furthermore, our method successfully compromises a commercial RAG system deployed on Alibaba's BaiLian platform, demonstrating its practical threat in real-world applications. These findings underscore the need for more robust and secure RAG frameworks to defend against poisoning attacks.

摘要: 检索增强生成（RAG）通过合并外部知识来增强大型语言模型（LLM），但其开放性引入了可被中毒攻击利用的漏洞。现有的RAG系统中毒方法存在局限性，例如概括性较差以及对抗性文本缺乏流畅性。在本文中，我们提出了CPA-RAG，这是一个黑盒对抗框架，可以生成与查询相关的文本，这些文本能够操纵检索过程以诱导目标答案。所提出的方法集成了基于文本的生成，通过多个LLM的交叉引导优化，以及基于检索器的评分来构建高质量的对抗样本。我们在多个数据集和LLM中进行了广泛的实验，以评估其有效性。结果表明，当top-k检索设置为5时，该框架的攻击成功率超过90%，与白盒性能相匹配，并在不同top-k值之间保持约5个百分点的一致优势。在各种防御策略下，它还比现有黑匣子基线高出14.5个百分点。此外，我们的方法成功地破坏了阿里巴巴百联平台上部署的商业RAG系统，证明了其在现实应用中的实际威胁。这些发现强调了需要更强大、更安全的RAG框架来抵御中毒攻击。



## **21. Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models**

越狱音频长凳：深入评估和分析大型音频语言模型的越狱威胁 cs.SD

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2501.13772v2) [paper-pdf](http://arxiv.org/pdf/2501.13772v2)

**Authors**: Hao Cheng, Erjia Xiao, Jing Shao, Yichi Wang, Le Yang, Chao Sheng, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Large Language Models (LLMs) demonstrate impressive zero-shot performance across a wide range of natural language processing tasks. Integrating various modality encoders further expands their capabilities, giving rise to Multimodal Large Language Models (MLLMs) that process not only text but also visual and auditory modality inputs. However, these advanced capabilities may also pose significant security risks, as models can be exploited to generate harmful or inappropriate content through jailbreak attacks. While prior work has extensively explored how manipulating textual or visual modality inputs can circumvent safeguards in LLMs and MLLMs, the vulnerability of audio-specific Jailbreak on Large Audio-Language Models (LALMs) remains largely underexplored. To address this gap, we introduce Jailbreak-AudioBench, which consists of the Toolbox, curated Dataset, and comprehensive Benchmark. The Toolbox supports not only text-to-audio conversion but also a range of audio editing techniques. The curated Dataset provides diverse explicit and implicit jailbreak audio examples in both original and edited forms. Utilizing this dataset, we evaluate multiple state-of-the-art LALMs, establishing the most comprehensive audio jailbreak benchmark to date. Finally, Jailbreak-AudioBench establishes a foundation for advancing future research on LALMs safety alignment by enabling the in-depth exposure of more powerful jailbreak threats, such as query-based audio editing, and by facilitating the development of effective defense mechanisms.

摘要: 大型语言模型（LLM）在广泛的自然语言处理任务中表现出令人印象深刻的零冲击性能。集成各种模式编码器进一步扩展了它们的功能，从而产生了多模式大型语言模型（MLLM），不仅处理文本，还处理视觉和听觉模式输入。然而，这些高级功能也可能带来重大的安全风险，因为模型可能会被利用来通过越狱攻击生成有害或不适当的内容。虽然之前的工作已经广泛探索了操纵文本或视觉模式输入如何规避LLM和MLLM中的保护措施，但大型音频语言模型（LALM）上的音频特定越狱的漏洞在很大程度上仍然没有得到充分的研究。为了弥补这一差距，我们引入了Jailbreak-AudioBench，它由收件箱、精心策划的数据集和全面的基准组成。收件箱不仅支持文本到音频转换，还支持一系列音频编辑技术。精心策划的数据集以原始和编辑的形式提供了多样化的显式和隐式越狱音频示例。利用该数据集，我们评估了多个最先进的LALM，建立了迄今为止最全面的音频越狱基准。最后，Jailbreak-AudioBench通过深入暴露更强大的越狱威胁（如基于查询的音频编辑），并促进有效防御机制的开发，为推进LALM安全对齐的未来研究奠定了基础。



## **22. QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language**

QuickAttack：使用结构化非自然查询语言越狱对齐的大型语言模型 cs.CR

To appear in ACL 2025

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2502.09723v3) [paper-pdf](http://arxiv.org/pdf/2502.09723v3)

**Authors**: Qingsong Zou, Jingyu Xiao, Qing Li, Zhi Yan, Yuhang Wang, Li Xu, Wenxuan Wang, Kuofeng Gao, Ruoyu Li, Yong Jiang

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable potential in the field of natural language processing. Unfortunately, LLMs face significant security and ethical risks. Although techniques such as safety alignment are developed for defense, prior researches reveal the possibility of bypassing such defenses through well-designed jailbreak attacks. In this paper, we propose QueryAttack, a novel framework to examine the generalizability of safety alignment. By treating LLMs as knowledge databases, we translate malicious queries in natural language into structured non-natural query language to bypass the safety alignment mechanisms of LLMs. We conduct extensive experiments on mainstream LLMs, and the results show that QueryAttack not only can achieve high attack success rates (ASRs), but also can jailbreak various defense methods. Furthermore, we tailor a defense method against QueryAttack, which can reduce ASR by up to $64\%$ on GPT-4-1106. Our code is available at https://github.com/horizonsinzqs/QueryAttack.

摘要: 大型语言模型（LLM）的最新进展在自然语言处理领域展示了巨大的潜力。不幸的是，LLM面临着巨大的安全和道德风险。尽管安全对齐等技术是为了防御而开发的，但之前的研究揭示了通过精心设计的越狱攻击绕过此类防御的可能性。在本文中，我们提出了一种新颖的框架，用于检查安全对齐的通用性。通过将LLM视为知识数据库，我们将自然语言中的恶意查询翻译为结构化非自然查询语言，以绕过LLM的安全对齐机制。我们在主流LLM上进行了广泛的实验，结果表明，CredyAttack不仅可以实现高的攻击成功率（SVR），还可以越狱各种防御方法。此外，我们还定制了一种针对SecureAttack的防御方法，该方法可以在GPT-4-1106上将ASB降低高达64美元。我们的代码可在https://github.com/horizonsinzqs/QueryAttack上获取。



## **23. What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs**

多枪攻击中真正重要的是什么？LLM长期背景脆弱性的实证研究 cs.CL

Accepted by ACL 2025

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19773v1) [paper-pdf](http://arxiv.org/pdf/2505.19773v1)

**Authors**: Sangyeop Kim, Yohan Lee, Yongwoo Song, Kimin Lee

**Abstract**: We investigate long-context vulnerabilities in Large Language Models (LLMs) through Many-Shot Jailbreaking (MSJ). Our experiments utilize context length of up to 128K tokens. Through comprehensive analysis with various many-shot attack settings with different instruction styles, shot density, topic, and format, we reveal that context length is the primary factor determining attack effectiveness. Critically, we find that successful attacks do not require carefully crafted harmful content. Even repetitive shots or random dummy text can circumvent model safety measures, suggesting fundamental limitations in long-context processing capabilities of LLMs. The safety behavior of well-aligned models becomes increasingly inconsistent with longer contexts. These findings highlight significant safety gaps in context expansion capabilities of LLMs, emphasizing the need for new safety mechanisms.

摘要: 我们通过多镜头越狱（MSJ）调查大型语言模型（LLM）中的长上下文漏洞。我们的实验利用高达128 K个令牌的上下文长度。通过对不同教学风格、镜头密度、主题和格式的各种多镜头攻击设置的综合分析，我们发现上下文长度是决定攻击有效性的主要因素。至关重要的是，我们发现成功的攻击并不需要精心制作的有害内容。即使是重复的镜头或随机的虚拟文本也可能规避模型安全措施，这表明LLM的长上下文处理能力存在根本限制。对齐良好的模型的安全行为随着更长的环境变得越来越不一致。这些发现突出了LLM在上下文扩展能力方面的重大安全差距，强调了新安全机制的必要性。



## **24. VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models**

VisURA：一种针对越狱多模式大型语言模型的视觉链推理攻击 cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19684v1) [paper-pdf](http://arxiv.org/pdf/2505.19684v1)

**Authors**: Bingrui Sima, Linhua Cong, Wenxuan Wang, Kun He

**Abstract**: The emergence of Multimodal Large Language Models (MLRMs) has enabled sophisticated visual reasoning capabilities by integrating reinforcement learning and Chain-of-Thought (CoT) supervision. However, while these enhanced reasoning capabilities improve performance, they also introduce new and underexplored safety risks. In this work, we systematically investigate the security implications of advanced visual reasoning in MLRMs. Our analysis reveals a fundamental trade-off: as visual reasoning improves, models become more vulnerable to jailbreak attacks. Motivated by this critical finding, we introduce VisCRA (Visual Chain Reasoning Attack), a novel jailbreak framework that exploits the visual reasoning chains to bypass safety mechanisms. VisCRA combines targeted visual attention masking with a two-stage reasoning induction strategy to precisely control harmful outputs. Extensive experiments demonstrate VisCRA's significant effectiveness, achieving high attack success rates on leading closed-source MLRMs: 76.48% on Gemini 2.0 Flash Thinking, 68.56% on QvQ-Max, and 56.60% on GPT-4o. Our findings highlight a critical insight: the very capability that empowers MLRMs -- their visual reasoning -- can also serve as an attack vector, posing significant security risks.

摘要: 多模式大型语言模型（MLRM）的出现通过集成强化学习和思想链（CoT）监督，实现了复杂的视觉推理能力。然而，虽然这些增强的推理能力可以提高性能，但它们也引入了新的且未充分研究的安全风险。在这项工作中，我们系统地研究了MLRM中高级视觉推理的安全影响。我们的分析揭示了一个基本的权衡：随着视觉推理的改进，模型变得更容易受到越狱攻击。出于这一关键发现的动机，我们引入了VisCRA（视觉链推理攻击），这是一种新颖的越狱框架，它利用视觉推理链绕过安全机制。VisCRA将有针对性的视觉注意力掩蔽与两阶段推理归纳策略相结合，以精确控制有害输出。大量的实验证明了VisCRA的显著有效性，在领先的闭源MLRM上实现了高攻击成功率：Gemini 2.0 Flash Thinking上为76.48%，QvQ-Max上为68.56%，GPT-4 o上为56.60%。我们的研究结果强调了一个关键的见解：赋予MLRM权力的能力（它们的视觉推理）本身也可以作为攻击载体，构成重大的安全风险。



## **25. Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors**

您的语言模型可以像人类一样秘密写作：对LLM生成的文本检测器的对比重述攻击 cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.15337v2) [paper-pdf](http://arxiv.org/pdf/2505.15337v2)

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.

摘要: 学术抄袭等大型语言模型（LLM）的滥用推动了识别LLM生成文本的检测器的发展。为了绕过这些检测器，出现了故意重写这些文本以逃避检测的重述攻击。尽管取得了成功，但现有方法需要大量的数据和计算预算来训练专门的解释器，并且当面对先进的检测算法时，它们的攻击功效会大大降低。为了解决这个问题，我们提出了\textBF{Co} ntrasive\textBF{P}araphrase \textBF{A}ttack（CoPA），这是一种免训练方法，可以使用现成的LLM有效地欺骗文本检测器。第一步是仔细编写指令，鼓励LLM生成更多类似人类的文本。尽管如此，我们观察到LLM固有的统计偏差仍然会导致一些生成的文本携带某些可以被检测器捕获的类似机器的属性。为了克服这个问题，CoPA构建了一个辅助的类似机器的单词分布，与LLM生成的类似人类的分布形成对比。通过在解码过程中从类人分布中减去类机器模式，CoPA能够生成文本检测器难以识别的句子。我们的理论分析表明了拟议攻击的优越性。大量实验验证了CoPA在各种场景中欺骗文本检测器的有效性。



## **26. Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models**

将小麦与谷壳分开：精调语言模型的安全重新对齐的事后方法 cs.CL

16 pages, 14 figures. Camera-ready for ACL2025 findings

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2412.11041v3) [paper-pdf](http://arxiv.org/pdf/2412.11041v3)

**Authors**: Di Wu, Xin Lu, Yanyan Zhao, Bing Qin

**Abstract**: Although large language models (LLMs) achieve effective safety alignment at the time of release, they still face various safety challenges. A key issue is that fine-tuning often compromises the safety alignment of LLMs. To address this issue, we propose a method named IRR (Identify, Remove, and Recalibrate for Safety Realignment) that performs safety realignment for LLMs. The core of IRR is to identify and remove unsafe delta parameters from the fine-tuned models, while recalibrating the retained ones. We evaluate the effectiveness of IRR across various datasets, including both full fine-tuning and LoRA methods. Our results demonstrate that IRR significantly enhances the safety performance of fine-tuned models on safety benchmarks, such as harmful queries and jailbreak attacks, while maintaining their performance on downstream tasks. The source code is available at: https://anonymous.4open.science/r/IRR-BD4F.

摘要: 尽管大型语言模型（LLM）在发布时实现了有效的安全一致，但它们仍然面临各种安全挑战。一个关键问题是微调通常会损害LLM的安全对齐。为了解决这个问题，我们提出了一种名为IRR（识别、删除和重新校准以实现安全重新对准）的方法，该方法为LLM执行安全重新对准。IRR的核心是从微调模型中识别并删除不安全的Delta参数，同时重新校准保留的参数。我们评估IRR在各种数据集的有效性，包括完全微调和LoRA方法。我们的结果表明，IRR显着增强了经过微调的模型在安全基准（例如有害查询和越狱攻击）上的安全性能，同时保持了其在下游任务上的性能。源代码可访问：https://anonymous.4open.science/r/IRR-BD4F。



## **27. Evaluating Robustness of Large Audio Language Models to Audio Injection: An Empirical Study**

评估大型音频语言模型对音频注入的稳健性：一项实证研究 cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19598v1) [paper-pdf](http://arxiv.org/pdf/2505.19598v1)

**Authors**: Guanyu Hou, Jiaming He, Yinhang Zhou, Ji Guo, Yitong Qiao, Rui Zhang, Wenbo Jiang

**Abstract**: Large Audio-Language Models (LALMs) are increasingly deployed in real-world applications, yet their robustness against malicious audio injection attacks remains underexplored. This study systematically evaluates five leading LALMs across four attack scenarios: Audio Interference Attack, Instruction Following Attack, Context Injection Attack, and Judgment Hijacking Attack. Using metrics like Defense Success Rate, Context Robustness Score, and Judgment Robustness Index, their vulnerabilities and resilience were quantitatively assessed. Experimental results reveal significant performance disparities among models; no single model consistently outperforms others across all attack types. The position of malicious content critically influences attack effectiveness, particularly when placed at the beginning of sequences. A negative correlation between instruction-following capability and robustness suggests models adhering strictly to instructions may be more susceptible, contrasting with greater resistance by safety-aligned models. Additionally, system prompts show mixed effectiveness, indicating the need for tailored strategies. This work introduces a benchmark framework and highlights the importance of integrating robustness into training pipelines. Findings emphasize developing multi-modal defenses and architectural designs that decouple capability from susceptibility for secure LALMs deployment.

摘要: 大型音频语言模型（LALM）越来越多地部署在现实世界的应用程序中，但它们针对恶意音频注入攻击的稳健性仍然没有得到充分的研究。本研究系统地评估了四种攻击场景中的五种主要LALM：音频干扰攻击、指令跟随攻击、上下文注入攻击和判断劫持攻击。使用防御成功率、上下文稳健性得分和判断稳健性指数等指标，量化评估了他们的脆弱性和弹性。实验结果显示模型之间的显着性能差异;没有一个模型在所有攻击类型中始终优于其他模型。恶意内容的位置严重影响攻击效果，特别是当放置在序列的开头时。预防以下能力和鲁棒性之间的负相关性表明，严格遵守指令的模型可能更容易受到影响，与安全一致的模型相比，阻力更大。此外，系统提示显示了混合的效果，表明需要量身定制的战略。这项工作引入了基准框架，并强调了将稳健性集成到训练管道中的重要性。研究结果强调开发多模式防御和架构设计，将能力与安全LALM部署的敏感性脱钩。



## **28. Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes**

如果你认识我，请保护我：保护特定面部身份免受Deepfakes的侵害 cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19582v1) [paper-pdf](http://arxiv.org/pdf/2505.19582v1)

**Authors**: Kaiqing Lin, Zhiyuan Yan, Ke-Yue Zhang, Li Hao, Yue Zhou, Yuzhen Lin, Weixiang Li, Taiping Yao, Shouhong Ding, Bin Li

**Abstract**: Securing personal identity against deepfake attacks is increasingly critical in the digital age, especially for celebrities and political figures whose faces are easily accessible and frequently targeted. Most existing deepfake detection methods focus on general-purpose scenarios and often ignore the valuable prior knowledge of known facial identities, e.g., "VIP individuals" whose authentic facial data are already available. In this paper, we propose \textbf{VIPGuard}, a unified multimodal framework designed to capture fine-grained and comprehensive facial representations of a given identity, compare them against potentially fake or similar-looking faces, and reason over these comparisons to make accurate and explainable predictions. Specifically, our framework consists of three main stages. First, fine-tune a multimodal large language model (MLLM) to learn detailed and structural facial attributes. Second, we perform identity-level discriminative learning to enable the model to distinguish subtle differences between highly similar faces, including real and fake variations. Finally, we introduce user-specific customization, where we model the unique characteristics of the target face identity and perform semantic reasoning via MLLM to enable personalized and explainable deepfake detection. Our framework shows clear advantages over previous detection works, where traditional detectors mainly rely on low-level visual cues and provide no human-understandable explanations, while other MLLM-based models often lack a detailed understanding of specific face identities. To facilitate the evaluation of our method, we built a comprehensive identity-aware benchmark called \textbf{VIPBench} for personalized deepfake detection, involving the latest 7 face-swapping and 7 entire face synthesis techniques for generation.

摘要: 在数字时代，保护个人身份免受Deepfake攻击变得越来越重要，尤其是对于面孔易于接触且经常成为攻击目标的名人和政治人物。大多数现有的Deepfake检测方法都专注于通用场景，并且经常忽略已知面部身份的宝贵先验知识，例如，其真实面部数据已经可用的“VIP个人”。在本文中，我们提出了\textBF{VIPGuard}，这是一个统一的多模式框架，旨在捕获给定身份的细粒度和全面的面部表示，将它们与潜在的虚假或相似的面部进行比较，并推理这些比较以做出准确且可解释的预测。具体来说，我们的框架由三个主要阶段组成。首先，微调多模式大型语言模型（MLLM）以学习详细和结构化的面部属性。其次，我们执行身份级别的辨别学习，使模型能够区分高度相似的面孔之间的细微差异，包括真实和虚假的变体。最后，我们引入了特定于用户的定制，其中我们对目标人脸身份的独特特征进行建模，并通过MLLM执行语义推理，以实现个性化和可解释的深度伪造检测。与之前的检测工作相比，我们的框架显示出明显的优势，传统的检测器主要依赖于低级视觉线索，并且不提供人类可理解的解释，而其他基于MLLM的模型通常缺乏对特定面部身份的详细了解。为了促进对我们的方法的评估，我们构建了一个名为\textBF{VIPBench}的全面身份感知基准，用于个性化深度伪造检测，其中涉及最新的7种面部交换和7种完整面部合成技术。



## **29. Robo-Troj: Attacking LLM-based Task Planners**

Robo-Troj：攻击基于LLM的任务规划器 cs.RO

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.17070v2) [paper-pdf](http://arxiv.org/pdf/2504.17070v2)

**Authors**: Mohaiminul Al Nahian, Zainab Altaweel, David Reitano, Sabbir Ahmed, Shiqi Zhang, Adnan Siraj Rakin

**Abstract**: Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.

摘要: 机器人需要任务规划方法来实现不仅仅需要个人行动的目标。最近，大型语言模型（LLM）在任务规划方面表现出令人印象深刻的性能。LLM可以使用行动和目标的描述生成分步解决方案。尽管基于LLM的任务规划取得了成功，但研究这些系统安全方面的研究有限。本文中，我们开发了Robo-Troj，这是针对基于LLM的任务规划器的第一个多触发后门攻击，这是这项工作的主要贡献。作为一种多触发攻击，Robo-Troj经过训练以适应机器人应用领域的多样性。例如，可以使用独特的触发词，例如，“herical”，激活特定的恶意行为，例如，厨房机器人上的割伤手。此外，我们还开发了一种优化方法来选择最有效的触发词。通过展示基于LLM的规划者的脆弱性，我们的目标是促进安全机器人系统的开发。



## **30. One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs**

一次性即可：将多回合攻击整合为LLM的高效单回合攻击 cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2503.04856v2) [paper-pdf](http://arxiv.org/pdf/2503.04856v2)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.

摘要: 我们引入了一种新颖的框架，用于将多轮对抗性“越狱”提示整合到单轮查询中，从而显着减少了大型语言模型（LLM）对抗性测试所需的手动负担。虽然多回合人类越狱已被证明具有很高的攻击成功率，但它们需要相当大的人力和时间。我们的多回合到单回合（M2 S）方法--连字符化、数字化和Python化--系统地将多回合对话重新格式化为结构化的单回合提示。尽管消除了迭代的来回相互作用，但这些提示仍然保留并经常增强对抗能力：在对多回合人类越狱（MHJ）数据集的广泛评估中，M2 S方法在几种最先进的LLM中实现了从70.6%到95.9%的攻击成功率。值得注意的是，单回合提示的性能比最初的多回合攻击高出17.5个百分点，同时平均将代币使用量减少一半以上。进一步的分析表明，将恶意请求嵌入到列举或类代码结构中利用了“上下文盲目性”，绕过了本地护栏和外部输入输出过滤器。通过将多回合对话转换为简洁的单回合提示，M2 S框架为大规模红色团队提供了可扩展的工具，并揭示了当代LLM防御中的关键弱点。



## **31. Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers**

三个意识，一个传奇：具有自适应堆叠密码的越狱大型推理模型 cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.16241v3) [paper-pdf](http://arxiv.org/pdf/2505.16241v3)

**Authors**: Viet-Anh Nguyen, Shiqian Zhao, Gia Dao, Runyi Hu, Yi Xie, Luu Anh Tuan

**Abstract**: Recently, Large Reasoning Models (LRMs) have demonstrated superior logical capabilities compared to traditional Large Language Models (LLMs), gaining significant attention. Despite their impressive performance, the potential for stronger reasoning abilities to introduce more severe security vulnerabilities remains largely underexplored. Existing jailbreak methods often struggle to balance effectiveness with robustness against adaptive safety mechanisms. In this work, we propose SEAL, a novel jailbreak attack that targets LRMs through an adaptive encryption pipeline designed to override their reasoning processes and evade potential adaptive alignment. Specifically, SEAL introduces a stacked encryption approach that combines multiple ciphers to overwhelm the models reasoning capabilities, effectively bypassing built-in safety mechanisms. To further prevent LRMs from developing countermeasures, we incorporate two dynamic strategies - random and adaptive - that adjust the cipher length, order, and combination. Extensive experiments on real-world reasoning models, including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the effectiveness of our approach. Notably, SEAL achieves an attack success rate of 80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant margin of 27.2%. Warning: This paper contains examples of inappropriate, offensive, and harmful content.

摘要: 最近，与传统的大型语言模型（LLM）相比，大型推理模型（LRM）表现出了更高的逻辑能力，引起了人们的广泛关注。尽管它们的性能令人印象深刻，但更强的推理能力引入更严重的安全漏洞的潜力在很大程度上仍然没有得到充分的探索。现有的越狱方法常常难以平衡有效性与鲁棒性与自适应安全机制。在这项工作中，我们提出了SEAL，这是一种新型越狱攻击，通过自适应加密管道针对LRM，该管道旨在覆盖它们的推理过程并规避潜在的自适应对齐。具体来说，SEAL引入了一种堆叠加密方法，该方法结合了多个密码来压倒模型的推理能力，有效地绕过了内置的安全机制。为了进一步防止LRM制定对策，我们结合了两种动态策略--随机和自适应--来调整密码长度、顺序和组合。对真实世界推理模型（包括DeepSeek-R1、Claude Sonnet和OpenAI GPT-o 4）的广泛实验验证了我们方法的有效性。值得注意的是，SEAL在GPT o 4-mini上的攻击成功率为80.8%，远远超过最先进的基线27.2%。警告：本文包含不恰当、冒犯性和有害内容的示例。



## **32. Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety**

良性样本很重要！对异常值良性样本进行微调严重破坏安全性 cs.LG

26 pages, 13 figures

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.06843v2) [paper-pdf](http://arxiv.org/pdf/2505.06843v2)

**Authors**: Zihan Guan, Mengxuan Hu, Ronghang Zhu, Sheng Li, Anil Vullikanti

**Abstract**: Recent studies have uncovered a troubling vulnerability in the fine-tuning stage of large language models (LLMs): even fine-tuning on entirely benign datasets can lead to a significant increase in the harmfulness of LLM outputs. Building on this finding, our red teaming study takes this threat one step further by developing a more effective attack. Specifically, we analyze and identify samples within benign datasets that contribute most to safety degradation, then fine-tune LLMs exclusively on these samples. We approach this problem from an outlier detection perspective and propose Self-Inf-N, to detect and extract outliers for fine-tuning. Our findings reveal that fine-tuning LLMs on 100 outlier samples selected by Self-Inf-N in the benign datasets severely compromises LLM safety alignment. Extensive experiments across seven mainstream LLMs demonstrate that our attack exhibits high transferability across different architectures and remains effective in practical scenarios. Alarmingly, our results indicate that most existing mitigation strategies fail to defend against this attack, underscoring the urgent need for more robust alignment safeguards. Codes are available at https://github.com/GuanZihan/Benign-Samples-Matter.

摘要: 最近的研究发现了大型语言模型（LLM）微调阶段的一个令人不安的漏洞：即使对完全良性的数据集进行微调也可能导致LLM输出的危害性显着增加。在这一发现的基础上，我们的红色团队研究通过开发更有效的攻击来进一步推进这一威胁。具体来说，我们分析和识别良性数据集中对安全性下降影响最大的样本，然后专门对这些样本进行微调。我们从异常值检测的角度来解决这个问题，并提出Self-Inf-N来检测和提取异常值以进行微调。我们的研究结果表明，对Self-Inf-N在良性数据集中选择的100个异常值样本进行微调LLM会严重损害LLM的安全性对齐。针对七种主流LLM的广泛实验表明，我们的攻击在不同架构中表现出高度的可移植性，并且在实际场景中仍然有效。令人震惊的是，我们的结果表明，大多数现有的缓解策略都无法抵御这种攻击，这凸显了迫切需要更强大的对齐保障措施。代码可访问https://github.com/GuanZihan/Benign-Samples-Matter。



## **33. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

具有事后感知校准的潜在空间对抗训练，用于保护大型语言模型免受越狱攻击 cs.CR

Under Review

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2501.10639v2) [paper-pdf](http://arxiv.org/pdf/2501.10639v2)

**Authors**: Xin Yi, Yue Li, dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment has become a critical requirement for large language models (LLMs), particularly given their widespread deployment in real-world applications. However, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to bypass safety measures and generate harmful outputs. Although numerous defense mechanisms based on adversarial training have been proposed, a persistent challenge lies in the exacerbation of over-refusal behaviors, which compromise the overall utility of the model. To address these challenges, we propose a Latent-space Adversarial Training with Post-aware Calibration (LATPC) framework. During the adversarial training phase, LATPC compares harmful and harmless instructions in the latent space and extracts safety-critical dimensions to construct refusal features attack, precisely simulating agnostic jailbreak attack types requiring adversarial mitigation. At the inference stage, an embedding-level calibration mechanism is employed to alleviate over-refusal behaviors with minimal computational overhead. Experimental results demonstrate that, compared to various defense methods across five types of jailbreak attacks, LATPC framework achieves a superior balance between safety and utility. Moreover, our analysis underscores the effectiveness of extracting safety-critical dimensions from the latent space for constructing robust refusal feature attacks.

摘要: 确保安全一致已成为大型语言模型（LLM）的关键要求，特别是考虑到它们在现实世界应用程序中的广泛部署。然而，LLM仍然容易受到越狱攻击，这些攻击利用系统漏洞绕过安全措施并产生有害输出。尽管已经提出了许多基于对抗训练的防御机制，但一个持续的挑战在于过度拒绝行为的加剧，这损害了该模型的整体实用性。为了应对这些挑战，我们提出了一种具有事后感知校准的潜在空间对抗训练（LAPC）框架。在对抗训练阶段，LAPC比较潜在空间中的有害和无害指令，并提取安全关键维度来构建拒绝特征攻击，精确模拟需要对抗缓解的不可知越狱攻击类型。在推理阶段，采用嵌入级校准机制以最小的计算负担减轻过度拒绝行为。实验结果表明，与五种越狱攻击的各种防御方法相比，LAPC框架在安全性和实用性之间实现了更好的平衡。此外，我们的分析强调了从潜在空间中提取安全关键维度以构建稳健拒绝特征攻击的有效性。



## **34. An Embarrassingly Simple Defense Against LLM Abliteration Attacks**

针对LLM删节攻击的令人尴尬的简单防御 cs.CL

preprint

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19056v1) [paper-pdf](http://arxiv.org/pdf/2505.19056v1)

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, Bernard Ghanem, George Turkiyyah

**Abstract**: Large language models (LLMs) are typically aligned to comply with safety guidelines by refusing harmful instructions. A recent attack, termed abliteration, isolates and suppresses the single latent direction most responsible for refusal behavior, enabling the model to generate unethical content. We propose a defense that modifies how models generate refusals. We construct an extended-refusal dataset that contains harmful prompts with a full response that justifies the reason for refusal. We then fine-tune Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on our extended-refusal dataset, and evaluate the resulting systems on a set of harmful prompts. In our experiments, extended-refusal models maintain high refusal rates, dropping at most by 10%, whereas baseline models' refusal rates drop by 70-80% after abliteration. A broad evaluation of safety and utility shows that extended-refusal fine-tuning neutralizes the abliteration attack while preserving general performance.

摘要: 大型语言模型（LLM）通常通过拒绝有害指令来调整以遵守安全指南。最近的一次攻击称为“取消”，它隔离和抑制了对拒绝行为最负责的单一潜在方向，使模型能够生成不道德的内容。我们提出了一种防御措施，可以修改模型生成拒绝的方式。我们构建了一个扩展拒绝数据集，其中包含有害提示，并给出了证明拒绝理由的完整回应。然后，我们在我们的扩展拒绝数据集上微调Llama-2- 7 B-Chat和Qwen2.5-Instruct（1.5B和3B参数），并在一组有害提示上评估结果系统。在我们的实验中，扩展的拒绝模型保持了较高的拒绝率，最多下降了10%，而基线模型的拒绝率下降了70-80%。安全性和实用性的广泛评估表明，扩展拒绝微调中和了abliteration攻击，同时保持了一般性能。



## **35. LLMs know their vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts**

LLM了解自己的弱点：通过自然分配转变揭露安全差距 cs.CL

ACL 2025 main conference. Code is available at  https://github.com/AI45Lab/ActorAttack

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2410.10700v2) [paper-pdf](http://arxiv.org/pdf/2410.10700v2)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: Safety concerns in large language models (LLMs) have gained significant attention due to their exposure to potentially harmful data during pre-training. In this paper, we identify a new safety vulnerability in LLMs: their susceptibility to \textit{natural distribution shifts} between attack prompts and original toxic prompts, where seemingly benign prompts, semantically related to harmful content, can bypass safety mechanisms. To explore this issue, we introduce a novel attack method, \textit{ActorBreaker}, which identifies actors related to toxic prompts within pre-training distribution to craft multi-turn prompts that gradually lead LLMs to reveal unsafe content. ActorBreaker is grounded in Latour's actor-network theory, encompassing both human and non-human actors to capture a broader range of vulnerabilities. Our experimental results demonstrate that ActorBreaker outperforms existing attack methods in terms of diversity, effectiveness, and efficiency across aligned LLMs. To address this vulnerability, we propose expanding safety training to cover a broader semantic space of toxic content. We thus construct a multi-turn safety dataset using ActorBreaker. Fine-tuning models on our dataset shows significant improvements in robustness, though with some trade-offs in utility. Code is available at https://github.com/AI45Lab/ActorAttack.

摘要: 由于大型语言模型（LLM）在预训练期间暴露于潜在有害的数据，因此其安全问题受到了广泛关注。在本文中，我们发现了LLM中的一个新安全漏洞：它们对攻击提示和原始有毒提示之间的\textit{自然分布变化}的敏感性，其中看似良性的提示，在语义上与有害内容相关，可以绕过安全机制。为了探索这个问题，我们引入了一种新颖的攻击方法\texttit {ActorBreaker}，它识别与预训练分发中有毒提示相关的参与者，以制作多回合提示，逐渐导致LLM揭露不安全内容。ActorBreaker以拉图尔的行为者网络理论为基础，涵盖人类和非人类行为者，以捕捉更广泛的漏洞。我们的实验结果表明，ActorBreaker在对齐LLM的多样性、有效性和效率方面优于现有的攻击方法。为了解决这个漏洞，我们建议扩大安全培训，以覆盖更广泛的有毒内容语义空间。因此，我们使用ActorBreaker构建了一个多圈安全数据集。我们数据集上的微调模型显示出鲁棒性的显着提高，但在效用方面存在一些权衡。代码可在https://github.com/AI45Lab/ActorAttack上获得。



## **36. GhostPrompt: Jailbreaking Text-to-image Generative Models based on Dynamic Optimization**

GhostPrompt：基于动态优化的越狱文本到图像生成模型 cs.LG

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.18979v1) [paper-pdf](http://arxiv.org/pdf/2505.18979v1)

**Authors**: Zixuan Chen, Hao Lin, Ke Xu, Xinghao Jiang, Tanfeng Sun

**Abstract**: Text-to-image (T2I) generation models can inadvertently produce not-safe-for-work (NSFW) content, prompting the integration of text and image safety filters. Recent advances employ large language models (LLMs) for semantic-level detection, rendering traditional token-level perturbation attacks largely ineffective. However, our evaluation shows that existing jailbreak methods are ineffective against these modern filters. We introduce GhostPrompt, the first automated jailbreak framework that combines dynamic prompt optimization with multimodal feedback. It consists of two key components: (i) Dynamic Optimization, an iterative process that guides a large language model (LLM) using feedback from text safety filters and CLIP similarity scores to generate semantically aligned adversarial prompts; and (ii) Adaptive Safety Indicator Injection, which formulates the injection of benign visual cues as a reinforcement learning problem to bypass image-level filters. GhostPrompt achieves state-of-the-art performance, increasing the ShieldLM-7B bypass rate from 12.5\% (Sneakyprompt) to 99.0\%, improving CLIP score from 0.2637 to 0.2762, and reducing the time cost by $4.2 \times$. Moreover, it generalizes to unseen filters including GPT-4.1 and successfully jailbreaks DALLE 3 to generate NSFW images in our evaluation, revealing systemic vulnerabilities in current multimodal defenses. To support further research on AI safety and red-teaming, we will release code and adversarial prompts under a controlled-access protocol.

摘要: 文本到图像（T2 I）生成模型可能会无意中产生不安全工作（NSFW）内容，从而促使文本和图像安全过滤器的集成。最近的进展使用大型语言模型（LLM）进行语义级检测，使传统的符号级扰动攻击基本上无效。然而，我们的评估表明，现有的越狱方法对这些现代过滤器无效。我们引入GhostPrompt，这是第一个自动越狱框架，将动态提示优化与多模式反馈相结合。它由两个关键组件组成：（i）动态优化，这是一个迭代过程，使用来自文本安全过滤器的反馈和CLIP相似性分数来指导大型语言模型（LLM）生成语义对齐的对抗提示;和（ii）自适应安全指标注入，它将良性视觉线索的注入制定为强化学习问题，以绕过图像级过滤器。GhostPrompt实现了最先进的性能，将ShieldLM-7 B旁路率从12.5%（Sneakypromit）提高到99.0%，将CLIP评分从0.2637提高到0.2762，并将时间成本减少4.2美元。此外，它还推广到了包括GPT-4.1在内的不可见过滤器，并在我们的评估中成功越狱DALLE 3以生成NSFW图像，揭示了当前多模式防御中的系统性漏洞。为了支持对人工智能安全和红色团队的进一步研究，我们将在受控访问协议下发布代码和对抗提示。



## **37. Exemplifying Emerging Phishing: QR-based Browser-in-The-Browser (BiTB) Attack**

示例新兴网络钓鱼：基于QR的浏览器中浏览器（BiTB）攻击 cs.CR

This manuscript is of 5 pages including 7 figures and 2 algorithms

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.18944v1) [paper-pdf](http://arxiv.org/pdf/2505.18944v1)

**Authors**: Muhammad Wahid Akram, Keshav Sood, Muneeb Ul Hassan, Basant Subba

**Abstract**: Lately, cybercriminals constantly formulate productive approaches to exploit individuals. This article exemplifies an innovative attack, namely QR-based Browser-in-The-Browser (BiTB), using proficiencies of Large Language Model (LLM) i.e. Google Gemini. The presented attack is a fusion of two emerging attacks: BiTB and Quishing (QR code phishing). Our study underscores attack's simplistic implementation utilizing malicious prompts provided to Gemini-LLM. Moreover, we presented a case study to highlight a lucrative attack method, we also performed an experiment to comprehend the attack execution on victims' device. The findings of this work obligate the researchers' contributions in confronting this type of phishing attempts through LLMs.

摘要: 最近，网络犯罪分子不断制定有效的方法来剥削个人。本文举例说明了一种创新性攻击，即基于QR的浏览器中浏览器（BiTB），使用了熟练的大型语言模型（LLM），即Google Gemini。所呈现的攻击是两种新兴攻击的融合：BiTB和Quishing（二维码网络钓鱼）。我们的研究强调了攻击利用向Gemini-LLM提供的恶意提示的简单化实施。此外，我们还提供了一个案例研究来强调一种有利可图的攻击方法，我们还进行了一项实验来了解对受害者设备的攻击执行情况。这项工作的发现要求研究人员通过LLM应对此类网络钓鱼企图做出贡献。



## **38. Stronger Enforcement of Instruction Hierarchy via Augmented Intermediate Representations**

通过增强的中间表示更强有力地执行指令层次结构 cs.AI

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.18907v1) [paper-pdf](http://arxiv.org/pdf/2505.18907v1)

**Authors**: Sanjay Kariyappa, G. Edward Suh

**Abstract**: Prompt injection attacks are a critical security vulnerability in large language models (LLMs), allowing attackers to hijack model behavior by injecting malicious instructions within the input context. Recent defense mechanisms have leveraged an Instruction Hierarchy (IH) Signal, often implemented through special delimiter tokens or additive embeddings to denote the privilege level of input tokens. However, these prior works typically inject the IH signal exclusively at the initial input layer, which we hypothesize limits its ability to effectively distinguish the privilege levels of tokens as it propagates through the different layers of the model. To overcome this limitation, we introduce a novel approach that injects the IH signal into the intermediate token representations within the network. Our method augments these representations with layer-specific trainable embeddings that encode the privilege information. Our evaluations across multiple models and training methods reveal that our proposal yields between $1.6\times$ and $9.2\times$ reduction in attack success rate on gradient-based prompt injection attacks compared to state-of-the-art methods, without significantly degrading the model's utility.

摘要: 提示注入攻击是大型语言模型（LLM）中的一个重要安全漏洞，允许攻击者通过在输入上下文中注入恶意指令来劫持模型行为。最近的防御机制利用了指令层次结构（IHS）信号，通常通过特殊的RST令牌或添加性嵌入来实现，以表示输入令牌的特权级别。然而，这些先前的作品通常只在初始输入层注入IHS信号，我们假设这限制了其在令牌通过模型的不同层传播时有效区分令牌特权级别的能力。为了克服这一限制，我们引入了一种新颖的方法，将HH信号注入网络内的中间令牌表示中。我们的方法通过编码特权信息的特定于层的可训练嵌入来增强这些表示。我们对多个模型和训练方法的评估显示，与最先进的方法相比，我们的提案可以使基于梯度的即时注入攻击的攻击成功率降低1.6美元到9.2美元，而不会显着降低模型的实用性。



## **39. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全性问题综述 cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18889v1) [paper-pdf](http://arxiv.org/pdf/2505.18889v1)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as GPT-4 (and its recent iterations like GPT-4o and the GPT-4.1 series), Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks (including input perturbations and data poisoning), misuse by malicious actors (e.g., for disinformation, phishing, and malware generation), and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives (scheming), which may even persist through safety training. We summarize recent academic and industrial studies (2022-2025) that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: 大型语言模型（LLM），例如GPT-4（及其最近的迭代，例如GPT-4 o和GPT-4.1系列）、谷歌的Gemini、Anthropic的Claude 3模型和xAI的Grok，已经引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。在本调查中，我们全面概述了LLM周围新出现的安全问题，将威胁分为即时注入和越狱、对抗性攻击（包括输入扰动和数据中毒）、恶意行为者的滥用（例如，虚假信息、网络钓鱼和恶意软件生成），以及自主LLM代理固有的令人担忧的风险。最近人们对后者给予了极大的关注，探索目标失调、紧急欺骗、自我保护本能，以及LLM制定和追求隐蔽、失调目标（阴谋）的潜力，甚至可能通过安全培训持续存在。我们总结了最近的学术和工业研究（2022-2025年），这些研究揭示了每种威胁，分析了拟议的防御措施及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **40. Audio Jailbreak Attacks: Exposing Vulnerabilities in SpeechGPT in a White-Box Framework**

音频越狱攻击：在白盒框架中暴露SpeechGPT中的漏洞 cs.CL

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18864v1) [paper-pdf](http://arxiv.org/pdf/2505.18864v1)

**Authors**: Binhao Ma, Hanqing Guo, Zhengping Jay Luo, Rui Duan

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced the naturalness and flexibility of human computer interaction by enabling seamless understanding across text, vision, and audio modalities. Among these, voice enabled models such as SpeechGPT have demonstrated considerable improvements in usability, offering expressive, and emotionally responsive interactions that foster deeper connections in real world communication scenarios. However, the use of voice introduces new security risks, as attackers can exploit the unique characteristics of spoken language, such as timing, pronunciation variability, and speech to text translation, to craft inputs that bypass defenses in ways not seen in text-based systems. Despite substantial research on text based jailbreaks, the voice modality remains largely underexplored in terms of both attack strategies and defense mechanisms. In this work, we present an adversarial attack targeting the speech input of aligned MLLMs in a white box scenario. Specifically, we introduce a novel token level attack that leverages access to the model's speech tokenization to generate adversarial token sequences. These sequences are then synthesized into audio prompts, which effectively bypass alignment safeguards and to induce prohibited outputs. Evaluated on SpeechGPT, our approach achieves up to 89 percent attack success rate across multiple restricted tasks, significantly outperforming existing voice based jailbreak methods. Our findings shed light on the vulnerabilities of voice-enabled multimodal systems and to help guide the development of more robust next-generation MLLMs.

摘要: 多模式大型语言模型（MLLM）的最新进展通过实现文本、视觉和音频模式的无缝理解，显着增强了人机交互的自然性和灵活性。其中，SpeechGPT等语音支持模型在可用性方面表现出了相当大的改进，提供了富有表达力和情感响应的交互，从而在现实世界通信场景中促进了更深层次的联系。然而，语音的使用会带来新的安全风险，因为攻击者可以利用口语的独特特征，例如时间、发音变异性以及语音到文本的翻译，以基于文本的系统中所没有的方式制作绕过防御的输入。尽管对基于文本的越狱进行了大量研究，但语音模式在攻击策略和防御机制方面仍然很大程度上没有得到充分的研究。在这项工作中，我们提出了一种针对白盒场景中对齐MLLM的语音输入的对抗攻击。具体来说，我们引入了一种新型的令牌级攻击，该攻击利用对模型的语音令牌化的访问来生成对抗性令牌序列。然后，这些序列被合成为音频提示，从而有效地绕过对齐保护措施并引发被禁止的输出。经过SpeechGPT评估，我们的方法在多个受限任务中实现了高达89%的攻击成功率，显着优于现有的基于语音的越狱方法。我们的研究结果揭示了语音多模式系统的漏洞，并帮助指导更强大的下一代MLLM的开发。



## **41. Strong Membership Inference Attacks on Massive Datasets and (Moderately) Large Language Models**

对海量数据集和（中等）大型语言模型的强成员推理攻击 cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18773v1) [paper-pdf](http://arxiv.org/pdf/2505.18773v1)

**Authors**: Jamie Hayes, Ilia Shumailov, Christopher A. Choquette-Choo, Matthew Jagielski, George Kaissis, Katherine Lee, Milad Nasr, Sahra Ghalebikesabi, Niloofar Mireshghallah, Meenatchi Sundaram Mutu Selva Annamalai, Igor Shilov, Matthieu Meeus, Yves-Alexandre de Montjoye, Franziska Boenisch, Adam Dziedzic, A. Feder Cooper

**Abstract**: State-of-the-art membership inference attacks (MIAs) typically require training many reference models, making it difficult to scale these attacks to large pre-trained language models (LLMs). As a result, prior research has either relied on weaker attacks that avoid training reference models (e.g., fine-tuning attacks), or on stronger attacks applied to small-scale models and datasets. However, weaker attacks have been shown to be brittle - achieving close-to-arbitrary success - and insights from strong attacks in simplified settings do not translate to today's LLMs. These challenges have prompted an important question: are the limitations observed in prior work due to attack design choices, or are MIAs fundamentally ineffective on LLMs? We address this question by scaling LiRA - one of the strongest MIAs - to GPT-2 architectures ranging from 10M to 1B parameters, training reference models on over 20B tokens from the C4 dataset. Our results advance the understanding of MIAs on LLMs in three key ways: (1) strong MIAs can succeed on pre-trained LLMs; (2) their effectiveness, however, remains limited (e.g., AUC<0.7) in practical settings; and, (3) the relationship between MIA success and related privacy metrics is not as straightforward as prior work has suggested.

摘要: 最先进的成员推理攻击（MIA）通常需要训练许多参考模型，因此很难将这些攻击扩展到大型预训练语言模型（LLM）。因此，先前的研究要么依赖于避免训练参考模型的较弱攻击（例如，微调攻击），或应用于小规模模型和数据集的更强攻击。然而，较弱的攻击已被证明是脆弱的-实现接近任意的成功-和见解，从强大的攻击在简化的设置不转化为今天的LLM。这些挑战引发了一个重要问题：在之前的工作中观察到的限制是由于攻击设计选择造成的，还是MIA对LLM从根本上无效？我们通过将LiRA（最强大的MIA之一）扩展到GPT-2架构，参数范围从10 M到1B，并在来自C4数据集中的超过20 B个令牌上训练参考模型来解决这个问题。我们的结果通过三个关键方式促进了对LLM上MIA的理解：（1）强大的MIA可以在预训练的LLM上取得成功;（2）然而，它们的有效性仍然有限（例如，在实际环境中，UC <0.7）;并且，（3）MIA成功与相关隐私指标之间的关系并不像之前的工作表明的那么简单。



## **42. Attacking Vision-Language Computer Agents via Pop-ups**

通过弹出窗口攻击视觉语言计算机代理 cs.CL

ACL 2025

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2411.02391v2) [paper-pdf](http://arxiv.org/pdf/2411.02391v2)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing their tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques, such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.

摘要: 由大型视觉和语言模型（VLM）支持的自治代理在完成日常计算机任务方面表现出了巨大的潜力，例如浏览网页预订旅行和操作桌面软件，这需要代理了解这些界面。尽管此类视觉输入越来越多地集成到代理应用程序中，但它们周围存在哪些类型的风险和攻击仍然不清楚。在这项工作中，我们证明了VLM代理很容易受到一组精心设计的对抗弹出窗口的攻击，而人类用户通常会识别并忽略这些弹出窗口。这种干扰导致特工单击这些弹出窗口，而不是像往常一样执行任务。将这些弹出窗口集成到现有的代理测试环境（如OSWorld和VisualWebArena）中，平均攻击成功率（代理单击弹出窗口的频率）为86%，任务成功率降低了47%。基本的防御技术，如要求代理忽略弹出窗口或包括广告通知，对攻击无效。



## **43. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

糖衣毒药：良性一代解锁法学硕士越狱 cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2504.05652v2) [paper-pdf](http://arxiv.org/pdf/2504.05652v2)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Hao Zhang, Jia-Chen Zhang, Zheng Zhou

**Abstract**: With the increasingly deep integration of large language models (LLMs) across diverse domains, the effectiveness of their safety mechanisms is encountering severe challenges. Currently, jailbreak attacks based on prompt engineering have become a major safety threat. However, existing methods primarily rely on black-box manipulation of prompt templates, resulting in poor interpretability and limited generalization. To break through the bottleneck, this study first introduces the concept of Defense Threshold Decay (DTD), revealing the potential safety impact caused by LLMs' benign generation: as benign content generation in LLMs increases, the model's focus on input instructions progressively diminishes. Building on this insight, we propose the Sugar-Coated Poison (SCP) attack paradigm, which uses a "semantic reversal" strategy to craft benign inputs that are opposite in meaning to malicious intent. This strategy induces the models to generate extensive benign content, thereby enabling adversarial reasoning to bypass safety mechanisms. Experiments show that SCP outperforms existing baselines. Remarkably, it achieves an average attack success rate of 87.23% across six LLMs. For defense, we propose Part-of-Speech Defense (POSD), leveraging verb-noun dependencies for syntactic analysis to enhance safety of LLMs while preserving their generalization ability.

摘要: 随着大型语言模型（LLM）跨不同领域的日益深入集成，其安全机制的有效性面临严峻挑战。目前，基于即时工程的越狱攻击已成为重大安全威胁。然而，现有的方法主要依赖于提示模板的黑匣子操作，导致可解释性较差且概括性有限。为了突破瓶颈，本研究首先引入了防御阈值衰变（DART）的概念，揭示了LLM良性生成对安全的潜在影响：随着LLM良性内容生成的增加，模型对输入指令的关注逐渐减少。基于这一见解，我们提出了糖衣毒药（SCP）攻击范式，该范式使用“语义逆转”策略来制造与恶意意图含义相反的良性输入。该策略促使模型生成广泛的良性内容，从而使对抗推理能够绕过安全机制。实验表明SCP优于现有基线。值得注意的是，它在六个LLM中的平均攻击成功率为87.23%。对于防御，我们提出了词性防御（POSD），利用动词-名词依赖进行语法分析，以增强LLM的安全性，同时保留其概括能力。



## **44. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

重温模型倒置评估：从误导性标准到可靠的隐私评估 cs.LG

To support future work, we release our MLLM-based MI evaluation  framework and benchmarking suite at  https://www.kaggle.com/datasets/hosytuyen/mi-reconstruction-collection

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.03519v3) [paper-pdf](http://arxiv.org/pdf/2505.03519v3)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework for such attacks relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI attacks and defenses without question. In this paper, we present the first in-depth study of this MI evaluation framework. In particular, we identify a critical issue of this standard MI evaluation framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by the target model T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 26 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation.

摘要: 模型反演（MI）攻击旨在通过利用对机器学习模型T的访问来从私有训练数据中重建信息。为了评估这种攻击，这种攻击的标准评估框架依赖于评估模型E，该模型E在与T相同的任务设计下训练。这个框架已经成为评估MI研究进展的事实上的标准，几乎在所有最近的MI攻击和防御中使用。在本文中，我们提出了第一个深入研究这个MI评估框架。特别是，我们确定了这个标准MI评估框架的一个关键问题：I型对抗性示例。这些重建没有捕获私有训练数据的视觉特征，但仍然被目标模型T认为是成功的，并最终可转移到E。这种假阳性损害了标准管理信息评价框架的可靠性。为了解决这个问题，我们引入了一个新的MI评估框架，用先进的多模态大型语言模型（MLLM）取代了评估模型E。通过利用他们的通用视觉理解，我们基于MLLM的框架不依赖于T中的共享任务设计的训练，从而降低了I型可移植性并提供对重建成功的更忠实评估。使用我们基于MLLM的评估框架，我们重新评估了26种不同的MI攻击设置，并根据经验揭示了标准评估框架下持续高的假阳性率。重要的是，我们证明了许多最先进的（SOTA）MI方法报告了夸大的攻击准确性，这表明实际的隐私泄露明显低于之前认为的。通过发现这一关键问题并提出强有力的解决方案，我们的工作能够重新评估MI研究的进展，并为可靠和强有力的评估设定新标准。



## **45. $PD^3F$: A Pluggable and Dynamic DoS-Defense Framework Against Resource Consumption Attacks Targeting Large Language Models**

$PD &#3F $：一个可插入且动态的DoS防御框架，针对针对大型语言模型的资源消耗攻击 cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18680v1) [paper-pdf](http://arxiv.org/pdf/2505.18680v1)

**Authors**: Yuanhe Zhang, Xinyue Wang, Haoran Gao, Zhenhong Zhou, Fanyu Meng, Yuyao Zhang, Sen Su

**Abstract**: Large Language Models (LLMs), due to substantial computational requirements, are vulnerable to resource consumption attacks, which can severely degrade server performance or even cause crashes, as demonstrated by denial-of-service (DoS) attacks designed for LLMs. However, existing works lack mitigation strategies against such threats, resulting in unresolved security risks for real-world LLM deployments. To this end, we propose the Pluggable and Dynamic DoS-Defense Framework ($PD^3F$), which employs a two-stage approach to defend against resource consumption attacks from both the input and output sides. On the input side, we propose the Resource Index to guide Dynamic Request Polling Scheduling, thereby reducing resource usage induced by malicious attacks under high-concurrency scenarios. On the output side, we introduce the Adaptive End-Based Suppression mechanism, which terminates excessive malicious generation early. Experiments across six models demonstrate that $PD^3F$ significantly mitigates resource consumption attacks, improving users' access capacity by up to 500% during adversarial load. $PD^3F$ represents a step toward the resilient and resource-aware deployment of LLMs against resource consumption attacks.

摘要: 由于大量的计算要求，大型语言模型（LLM）很容易受到资源消耗攻击，这可能会严重降低服务器性能，甚至导致崩溃，正如为LLM设计的拒绝服务（DPS）攻击所证明的那样。然而，现有作品缺乏针对此类威胁的缓解策略，导致现实世界LLM部署存在未解决的安全风险。为此，我们提出了可插入和动态DoS防御框架（$PD & 3F $），该框架采用两阶段方法来防御来自输入和输出端的资源消耗攻击。在输入端，我们提出了资源索引来指导动态请求投票调度，从而减少高并发场景下恶意攻击引发的资源使用。在输出端，我们引入了自适应基于端的抑制机制，该机制可以提前终止过多的恶意生成。六种模型的实验表明，$PD ' 3F $显着减轻了资源消耗攻击，在对抗负载期间将用户的访问能力提高了高达500%。$PD ' 3F $代表着向针对资源消耗攻击的弹性和资源感知部署LLM迈出了一步。



## **46. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**

LLM水印能否强大地防止未经授权的知识提炼？ cs.CL

Accepted by ACL 2025 (Main)

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2502.11598v2) [paper-pdf](http://arxiv.org/pdf/2502.11598v2)

**Authors**: Leyi Pan, Aiwei Liu, Shiyu Huang, Yijian Lu, Xuming Hu, Lijie Wen, Irwin King, Philip S. Yu

**Abstract**: The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.

摘要: 大语言模型（LLM）水印的放射性性质使得能够检测学生模型在基于带水印的教师模型的输出进行训练时继承的水印，使其成为防止未经授权的知识提炼的有希望的工具。然而，水印放射性对敌对行为者的稳健性在很大程度上仍然没有被探索。本文探讨学生模型是否可以通过知识提炼获得教师模型的能力，同时避免水印继承。我们提出了两类水印去除方法：通过非目标和目标训练数据重述（UP和TP）进行蒸馏前去除，以及通过推断时水印中和（WN）进行蒸馏后去除。跨多个模型对、水印方案和超参数设置的大量实验表明，TP和WN都可以彻底消除继承水印，WN在实现这一目标的同时保持知识传输效率和低计算负担。鉴于生产LLM中水印技术的持续部署，这些发现强调了对更强大的防御策略的迫切需要。我们的代码可在https://github.com/THU-BPM/Watermark-Radioactivity-Attack上获取。



## **47. Safety Alignment via Constrained Knowledge Unlearning**

通过受约束的知识忘记学习实现安全一致 cs.CL

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18588v1) [paper-pdf](http://arxiv.org/pdf/2505.18588v1)

**Authors**: Zesheng Shi, Yucheng Zhou, Jing Li

**Abstract**: Despite significant progress in safety alignment, large language models (LLMs) remain susceptible to jailbreak attacks. Existing defense mechanisms have not fully deleted harmful knowledge in LLMs, which allows such attacks to bypass safeguards and produce harmful outputs. To address this challenge, we propose a novel safety alignment strategy, Constrained Knowledge Unlearning (CKU), which focuses on two primary objectives: knowledge localization and retention, and unlearning harmful knowledge. CKU works by scoring neurons in specific multilayer perceptron (MLP) layers to identify a subset U of neurons associated with useful knowledge. During the unlearning process, CKU prunes the gradients of neurons in U to preserve valuable knowledge while effectively mitigating harmful content. Experimental results demonstrate that CKU significantly enhances model safety without compromising overall performance, offering a superior balance between safety and utility compared to existing methods. Additionally, our analysis of neuron knowledge sensitivity across various MLP layers provides valuable insights into the mechanics of safety alignment and model knowledge editing.

摘要: 尽管在安全调整方面取得了重大进展，但大型语言模型（LLM）仍然容易受到越狱攻击。现有的防御机制尚未完全删除LLM中的有害知识，这使得此类攻击绕过保障措施并产生有害输出。为了应对这一挑战，我们提出了一种新型的安全调整策略，即约束知识取消学习（CKU），它专注于两个主要目标：知识本地化和保留，以及取消有害知识。CKU通过对特定多层感知器（MLP）层中的神经元进行评分，以识别与有用知识相关的神经元子集U。在去学习过程中，CKU修剪U中神经元的梯度，以保留有价值的知识，同时有效地减少有害内容。实验结果表明，与现有方法相比，CKU显着增强了模型的安全性，在不影响整体性能的情况下提供了安全性和实用性之间的卓越平衡。此外，我们对各个MLP层的神经元知识敏感性的分析为安全对齐和模型知识编辑的机制提供了宝贵的见解。



## **48. MASTER: Multi-Agent Security Through Exploration of Roles and Topological Structures -- A Comprehensive Framework**

MASTER：通过角色和布局结构探索的多智能体安全--一个全面的框架 cs.MA

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18572v1) [paper-pdf](http://arxiv.org/pdf/2505.18572v1)

**Authors**: Yifan Zhu, Chao Zhang, Xin Shi, Xueqiao Zhang, Yi Yang, Yawei Luo

**Abstract**: Large Language Models (LLMs)-based Multi-Agent Systems (MAS) exhibit remarkable problem-solving and task planning capabilities across diverse domains due to their specialized agentic roles and collaborative interactions. However, this also amplifies the severity of security risks under MAS attacks. To address this, we introduce MASTER, a novel security research framework for MAS, focusing on diverse Role configurations and Topological structures across various scenarios. MASTER offers an automated construction process for different MAS setups and an information-flow-based interaction paradigm. To tackle MAS security challenges in varied scenarios, we design a scenario-adaptive, extensible attack strategy utilizing role and topological information, which dynamically allocates targeted, domain-specific attack tasks for collaborative agent execution. Our experiments demonstrate that such an attack, leveraging role and topological information, exhibits significant destructive potential across most models. Additionally, we propose corresponding defense strategies, substantially enhancing MAS resilience across diverse scenarios. We anticipate that our framework and findings will provide valuable insights for future research into MAS security challenges.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）由于其专业的代理角色和协作交互，在不同领域表现出出色的问题解决和任务规划能力。然而，这也放大了MAS攻击下安全风险的严重性。为了解决这个问题，我们引入了MASTER，这是一个针对MAS的新型安全研究框架，重点关注各种场景中的多样化角色配置和布局结构。MASTER为不同的MAS设置和基于信息流的交互范式提供了自动化构建流程。为了应对不同场景下的MAS安全挑战，我们利用角色和拓扑信息设计了一种自适应、可扩展的攻击策略，该策略为协作代理执行动态分配有针对性的、特定领域的攻击任务。我们的实验表明，这种利用角色和拓扑信息的攻击在大多数模型中表现出显着的破坏潜力。此外，我们还提出了相应的防御策略，大幅增强MAS在不同场景下的弹性。我们预计我们的框架和调查结果将为未来对MAS安全挑战的研究提供有价值的见解。



## **49. Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation**

通过意图操纵探索大型语言模型中内容审核保护的脆弱性 cs.CL

Preprint, under review. TL;DR: We propose a new two-stage  intent-based prompt-refinement framework, IntentPrompt, that aims to explore  the vulnerability of LLMs' content moderation guardrails by refining prompts  into benign-looking declarative forms via intent manipulation for red-teaming  purposes

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18556v1) [paper-pdf](http://arxiv.org/pdf/2505.18556v1)

**Authors**: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby G. Dagher, Haohan Wang

**Abstract**: Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails.

摘要: 意图检测是自然语言理解的核心组成部分，已大大发展成为保护大型语言模型（LLM）的关键机制。虽然之前的工作已应用意图检测来增强LLM的审核护栏，在对抗内容级越狱方面取得了显着成功，但这些意图感知护栏在恶意操纵下的稳健性仍然没有得到充分的探索。在这项工作中，我们调查了意图感知护栏的漏洞，并证明LLM具有隐式意图检测能力。我们提出了一个两阶段基于意图的预算细化框架IntentPrompt，该框架首先将有害的询问转化为结构化的大纲，并通过反馈循环迭代优化提示，进一步将其重新构建为宣言式叙事，以增强越狱成功率。红色团队的目的。针对四个公共基准测试和各种黑匣子LLM的广泛实验表明，我们的框架始终优于几种尖端的越狱方法，甚至可以规避高级意图分析（IA）和基于思想链（CoT）的防御。具体来说，我们的“FTR +SPIN”变体在o 1模型上针对基于CoT的防御的攻击成功率从88.25%到96.54%不等，在基于IA的防御下，在GPT-4 o模型上的攻击成功率从86.75%到97.12%不等。这些发现凸显了LLM安全机制的一个严重弱点，并表明意图操纵对内容审核护栏构成了越来越大的挑战。



## **50. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2407.20361v4) [paper-pdf](http://arxiv.org/pdf/2407.20361v4)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models - Stack model, VisualPhishNet, and Phishpedia - against PhishOracle-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a multimodal large language model (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether PhishOracle-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大网络安全威胁。机器学习（ML）和深度学习（DL）的进步导致了众多网络钓鱼网页检测解决方案的开发，但这些模型仍然容易受到对抗性攻击。评估其针对对抗性网络钓鱼网页的稳健性至关重要。现有工具包含为有限数量品牌预先设计的网络钓鱼网页数据集，并且网络钓鱼功能缺乏多样性。   为了应对这些挑战，我们开发了PhishOracle，这是一种通过将各种网络钓鱼功能嵌入到合法网页中来生成对抗性网络钓鱼网页的工具。我们评估了三个现有的特定任务模型（Stack模型、Visual PhishNet和Phishpedia）针对PhishOracle生成的对抗性网络钓鱼网页的稳健性，并观察到它们的检测率显着下降。相比之下，基于多模式大型语言模型（MLLM）的网络钓鱼检测器对这些对抗性攻击表现出更强的鲁棒性，但仍然容易规避。我们的研究结果强调了网络钓鱼检测模型对对抗性攻击的脆弱性，强调了对更强大的检测方法的需求。此外，我们还进行了一项用户研究，以评估PhishOracle生成的对抗性网络钓鱼网页是否会欺骗用户。结果表明，许多网络钓鱼网页不仅逃避了现有的检测模型，还逃避了用户。



