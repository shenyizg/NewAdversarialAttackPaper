# Latest Large Language Model Attack Papers
**update at 2025-07-16 09:45:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems**

跨域多代理LLM系统必须解决的七个安全挑战 cs.CR

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2505.23847v3) [paper-pdf](http://arxiv.org/pdf/2505.23847v3)

**Authors**: Ronny Ko, Jiseong Jeong, Shuyuan Zheng, Chuan Xiao, Tae-Wan Kim, Makoto Onizuka, Won-Yong Shin

**Abstract**: Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.

摘要: 大型语言模型（LLM）正在迅速演变为跨组织边界合作的自治代理，实现联合灾难响应、供应链优化以及其他需要分散专业知识而不放弃数据所有权的任务。然而，跨域协作打破了当前对齐和遏制技术背后的统一信任假设。孤立的良性代理在从不受信任的对等点接收消息时可能会泄露秘密或违反政策，从而产生由紧急多代理动态而不是经典软件错误驱动的风险。本立场文件绘制了跨域多代理LLM系统的安全议程。我们介绍了七类新型安全挑战，我们还针对每类挑战提供了合理的攻击、安全评估指标和未来的研究指南。



## **2. A Generative Approach to LLM Harmfulness Detection with Special Red Flag Tokens**

使用特殊红旗令牌进行LLM危害检测的生成方法 cs.CL

14 pages, 6 figures

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2502.16366v3) [paper-pdf](http://arxiv.org/pdf/2502.16366v3)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mofakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) are based on fine-tuning that forces models to shift from an unsafe answer to refusal when faced with harmful requests. Unfortunately, these drastic distribution shifts generally compromise model capabilities. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to train the model to insert this token into its response at any time when harmful content is generated or about to be generated. Our approach offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer and provides robustness as good as adversarial training without the need to run attacks during training. Moreover, by encapsulating our safety tuning in a LoRA module, we provide additional defenses against fine-tuning API attacks.

摘要: 大型语言模型（LLM）的大多数安全训练方法都基于微调，迫使模型在面临有害请求时从不安全的答案转向拒绝。不幸的是，这些急剧的分布变化通常会损害模型的能力。为了避免这种情况，我们建议使用一个我们称为红旗令牌（）的特殊令牌来扩展模型的词汇表<rf>，并建议训练模型，以便在生成或即将生成有害内容时随时将此令牌插入到其响应中。我们的方法提供了几个优点：它使模型能够明确地学习危害性的概念，同时对生成的分布产生轻微影响，从而保持模型的效用。它还评估每个生成的答案，并提供与对抗训练一样好的鲁棒性，而无需在训练期间运行攻击。此外，通过将我们的安全调优封装在LoRA模块中，我们提供了针对微调API攻击的额外防御。



## **3. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

GUARD：神经代码生成中基于双智能体的思维链后门防御 cs.SE

Accepted by SEKE 2025

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2505.21425v2) [paper-pdf](http://arxiv.org/pdf/2505.21425v2)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.

摘要: 随着大型语言模型在代码生成中的广泛应用，最近的研究表明，采用额外的思想链生成模型可以通过提供显式推理步骤来显着提高代码生成性能。然而，作为外部组件，CoT模型特别容易受到后门攻击，而现有的防御机制往往无法有效检测到后门攻击。为了应对这一挑战，我们提出了GUARD，这是一种新型双代理防御框架，专门设计用于对抗神经代码生成中的CoT后门攻击。GUARD集成了两个核心组件：GUARD-Judge，通过全面分析识别可疑的CoT步骤和潜在触发因素，以及GUARD-Repair，采用检索增强生成方法来为识别的异常重新生成安全CoT步骤。实验结果表明，GUARD有效地缓解了攻击，同时保持生成质量，推进了安全代码生成系统。



## **4. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

多触发中毒放大了LLM中的后门漏洞 cs.CL

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11112v1) [paper-pdf](http://arxiv.org/pdf/2507.11112v1)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.

摘要: 最近的研究表明，大型语言模型（LLM）容易受到数据中毒攻击，其中恶意训练示例嵌入了由特定输入模式触发的隐藏行为。然而，大多数现有的作品假设一个短语，并专注于攻击的有效性，提供有限的理解触发机制和多个触发器如何在模型中相互作用。在本文中，我们提出了一个框架，研究中毒的LLM。我们发现，多个不同的后门触发器可以共存于一个单一的模型中，而不会相互干扰，使对手能够同时嵌入多个触发器。使用具有高嵌入相似性的多个触发器，我们证明即使令牌被长令牌跨度替换或分开，中毒触发器也可以实现稳健的激活。我们的研究结果揭示了LLC中更广泛、更持久的脆弱性表面。为了减轻这种威胁，我们提出了一种事后恢复方法，该方法根据分层权重差异分析选择性地重新训练特定的模型组件。我们的方法通过最少的参数更新有效地消除了触发行为，从而提供了针对多触发中毒的实用有效防御。



## **5. The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs**

面具背后的魔鬼：扩散LLC的紧急安全漏洞 cs.CL

21 pages, 9 figures, work in progress

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11097v1) [paper-pdf](http://arxiv.org/pdf/2507.11097v1)

**Authors**: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang

**Abstract**: Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.

摘要: 基于扩散的大型语言模型（dLLM）最近成为自回归LLM的强大替代方案，通过并行解码和双向建模提供更快的推理和更强的交互性。然而，尽管在代码生成和文本填充方面表现出色，但我们发现了一个基本的安全问题：现有的对齐机制未能保护dLLM免受上下文感知、屏蔽输入对抗提示的影响，从而暴露了新颖的漏洞。为此，我们提出了DIJA，这是第一个利用dLLM独特安全弱点的系统性研究和越狱攻击框架。具体来说，我们提出的DIJA构建了对抗性交错屏蔽文本提示，利用dLLM的文本生成机制，即双向建模和并行解码。双向建模驱动模型为掩蔽跨度生成上下文一致的输出，即使是有害的，而并行解码限制了模型动态过滤和不安全内容的拒绝采样。这会导致标准对齐机制失败，从而导致在经过优化的DLLM中进行有害的完成，即使在提示中直接暴露了有害行为或不安全的指令。通过全面的实验，我们证明DIJA的性能显着优于现有的越狱方法，暴露了dLLM架构中之前被忽视的威胁表面。值得注意的是，我们的方法在Dream-Direct上实现了高达100%的基于关键词的ASB，超过了最强的先前基线ReNeLLM，在JailbreakBench上基于评估者的ASB中提高了高达78.5%，在StrongRESYS评分中提高了37.7分，同时不需要重写或隐藏越狱提示中的有害内容。我们的研究结果强调了重新思考这类新兴语言模型中的安全一致的迫切需要。代码可在www.example.com上获得。



## **6. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2504.01550v3) [paper-pdf](http://arxiv.org/pdf/2504.01550v3)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已经成为强大的工具，但其固有的安全风险-从有害内容生成到更广泛的社会危害-构成了重大挑战。最近的对抗攻击、微调漏洞以及在高风险环境中增加部署LLM可能会放大这些风险。现有的安全增强技术，例如通过人工反馈或对抗性训练进行微调，仍然很脆弱，因为它们解决了特定的威胁，并且通常无法概括看不见的攻击，或者需要手动系统级防御。本文介绍了RepBend，这是一种新颖的方法，它从根本上破坏了LLM中有害行为的潜在表现，提供了可扩展的解决方案来增强（潜在固有的）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **7. From Alerts to Intelligence: A Novel LLM-Aided Framework for Host-based Intrusion Detection**

从警报到情报：基于主机的入侵检测的新型LLM辅助框架 cs.CR

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.10873v1) [paper-pdf](http://arxiv.org/pdf/2507.10873v1)

**Authors**: Danyu Sun, Jinghuai Zhang, Jiacen Xu, Yu Zheng, Yuan Tian, Zhou Li

**Abstract**: Host-based intrusion detection system (HIDS) is a key defense component to protect the organizations from advanced threats like Advanced Persistent Threats (APT). By analyzing the fine-grained logs with approaches like data provenance, HIDS has shown successes in capturing sophisticated attack traces. Despite the progresses embarked by the research community and industry, HIDS still frequently encounters backlash from their operators in the deployed environments, due to issues like high false-positive rate, inconsistent outcomes across environments and human-unfriendly detection results. Large Language Models (LLMs) have great potentials to advance the state of HIDS, given their extensive knowledge of attack techniques and their ability to detect anomalies through semantic analysis, anchored by recent studies. Yet, our preliminary analysis indicates that building an HIDS by naively prompting an LLM is unlikely to succeed. In this work, we explore the direction of building a customized LLM pipeline for HIDS and develop a system named SHIELD. SHIELD addresses challenges related to LLM's token limits, confusion of background noises, etc., by integrating a variety of techniques like event-level Masked Autoencoder (MAE) for attack window detection, attack evidence identification and expansion, Deterministic Data Augmentation (DDA) for profiling normal activities, and multi-purpose prompting that guides the LLM to conduct precise and interpretable attack investigations. Extensive experiments on three log datasets (DARPA-E3, NodLink-simulated-data and ATLASv2) show that SHIELD consistently achieves outstanding performance in comparison with 5 representative HIDS. These findings highlight the potential of LLMs as powerful tools for intrusion detection and pave the way for future research in this domain.

摘要: 基于主机的入侵检测系统（HIDS）是保护组织免受高级持续威胁（APT）等高级威胁的关键防御组件。通过使用数据来源等方法分析细粒度日志，HIDS在捕获复杂的攻击痕迹方面取得了成功。尽管研究界和行业取得了进展，但由于假阳性率高、环境中结果不一致以及检测结果对人类不友好等问题，HIDS在部署环境中仍然经常遭到操作员的强烈反对。大型语言模型（LLM）在推进HIDS状态方面拥有巨大的潜力，因为它们对攻击技术有着广泛的了解，并且能够通过最近的研究得出的语义分析来检测异常。然而，我们的初步分析表明，通过天真地推动LLM来构建HIDS不太可能成功。在这项工作中，我们探索了为HIDS构建定制LLM管道的方向，并开发了一个名为SHIELD的系统。SHIELD解决了与LLM代币限制、背景噪音混乱等相关的挑战，通过集成各种技术，例如用于攻击窗口检测、攻击证据识别和扩展的事件级屏蔽自动编码器（MAE）、用于分析正常活动的确定性数据增强（DDA）以及指导LLM进行精确且可解释的攻击调查的多用途提示。在DARPA-E3、NodLink模拟数据和ATLASv 2三个日志数据集上进行的大量实验表明，SHIELD与5个代表性的HIDS相比，表现出了优异的性能。这些发现突出了LLM作为入侵检测的强大工具的潜力，并为该领域的未来研究铺平了道路。



## **8. REAL-IoT: Characterizing GNN Intrusion Detection Robustness under Practical Adversarial Attack**

REAL-IOT：描述实际对抗攻击下GNN入侵检测的鲁棒性 cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10836v1) [paper-pdf](http://arxiv.org/pdf/2507.10836v1)

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi

**Abstract**: Graph Neural Network (GNN)-based network intrusion detection systems (NIDS) are often evaluated on single datasets, limiting their ability to generalize under distribution drift. Furthermore, their adversarial robustness is typically assessed using synthetic perturbations that lack realism. This measurement gap leads to an overestimation of GNN-based NIDS resilience. To address the limitations, we propose \textbf{REAL-IoT}, a comprehensive framework for robustness evaluation of GNN-based NIDS in IoT environments. Our framework presents a methodology that creates a unified dataset from canonical datasets to assess generalization under drift. In addition, it features a novel intrusion dataset collected from a physical IoT testbed, which captures network traffic and attack scenarios under real-world settings. Furthermore, using REAL-IoT, we explore the usage of Large Language Models (LLMs) to analyze network data and mitigate the impact of adversarial examples by filtering suspicious flows. Our evaluations using REAL-IoT reveal performance drops in GNN models compared to results from standard benchmarks, quantifying their susceptibility to drift and realistic attacks. We also demonstrate the potential of LLM-based filtering to enhance robustness. These findings emphasize the necessity of realistic threat modeling and rigorous measurement practices for developing resilient IoT intrusion detection systems.

摘要: 基于图形神经网络（GNN）的网络入侵检测系统（NIDS）通常在单个数据集上进行评估，这限制了它们在分布漂移下进行概括的能力。此外，它们的对抗鲁棒性通常是使用缺乏真实性的合成扰动来评估的。这种测量差距导致高估了基于GNN的NIDS弹性。为了解决这些局限性，我们提出了\textBF{REAL-IoT}，这是一个用于在物联网环境中对基于GNN的NIDS进行稳健性评估的综合框架。我们的框架提出了一种方法，该方法从规范数据集创建统一数据集，以评估漂移下的概括性。此外，它还具有从物理物联网测试台收集的新型入侵数据集，该数据集可以捕获现实世界环境下的网络流量和攻击场景。此外，使用REAL-IOT，我们探索使用大型语言模型（LLM）来分析网络数据并通过过滤可疑流来减轻对抗示例的影响。我们使用REAL-IOT进行的评估显示，与标准基准的结果相比，GNN模型的性能有所下降，量化了它们对漂移和现实攻击的敏感性。我们还展示了基于LLM的过滤增强鲁棒性的潜力。这些发现强调了开发弹性物联网入侵检测系统的现实威胁建模和严格测量实践的必要性。



## **9. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

模型篡改攻击能够更严格地评估LLM能力 cs.CR

Accepted to TMLR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2502.05209v3) [paper-pdf](http://arxiv.org/pdf/2502.05209v3)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.

摘要: 对大型语言模型（LLM）风险和能力的评估越来越多地被纳入人工智能风险管理和治理框架中。目前，大多数风险评估都是通过设计从系统中引发有害行为的输入来进行的。然而，这种方法有两个局限性。首先，投入产出评估无法完全评估开权模型的现实风险。其次，在任何特定的投入-产出评估期间识别的行为只能下限模型的最坏可能情况的投入-产出行为。作为引发有害行为的补充方法，我们建议使用模型篡改攻击来评估LLM，该攻击允许修改潜在激活或权重。我们使用最先进的技术来消除有害的LLM功能，以对抗一系列5个输入空间和6个模型篡改攻击。除了对这些方法进行比较之外，我们还表明：（1）模型对能力启发攻击的弹性取决于低维鲁棒性子空间;（2）模型篡改攻击的成功率可以根据经验预测并为保持的输入空间攻击的成功提供保守估计;（3）最先进的取消学习方法可以在16个微调步骤内轻松取消。总之，这些结果突出了抑制有害LLM能力的困难，并表明模型篡改攻击比单独的输入空间攻击能够进行更严格的评估。



## **10. Logic layer Prompt Control Injection (LPCI): A Novel Security Vulnerability Class in Agentic Systems**

逻辑层提示控制注入（LCI）：统计系统中的一种新型安全漏洞类 cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10457v1) [paper-pdf](http://arxiv.org/pdf/2507.10457v1)

**Authors**: Hammad Atta, Ken Huang, Manish Bhatt, Kamal Ahmed, Muhammad Aziz Ul Haq, Yasir Mehmood

**Abstract**: The integration of large language models (LLMs) into enterprise systems has created a new class of covert security vulnerabilities, particularly within logic-execution layers and persistent-memory contexts. In this paper, we introduce Logic-Layer Prompt Control Injection (LPCI), a novel attack category in which encoded, delayed, and conditionally triggered payloads are embedded in memory, vector stores, or tool outputs. These payloads can bypass conventional input filters and trigger unauthorised behaviour across sessions.

摘要: 大型语言模型（LLM）集成到企业系统中产生了一类新的隐蔽安全漏洞，特别是在逻辑执行层和持久性内存上下文中。本文中，我们介绍了逻辑层提示控制注入（LPCI），这是一种新型攻击类别，其中编码、延迟和条件触发的有效负载嵌入到内存、载体存储或工具输出中。这些有效负载可以绕过传统的输入过滤器并触发跨会话的未经授权的行为。



## **11. Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems**

破解LLM护栏：即时注入和越狱检测系统规避攻击的实证分析 cs.CR

14 pages, 5 figures, 11 tables. To be published in LLMSec 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2504.11168v3) [paper-pdf](http://arxiv.org/pdf/2504.11168v3)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.

摘要: 大型语言模型（LLM）护栏系统旨在防止即时注入和越狱攻击。然而，他们仍然容易受到逃避技术的影响。我们演示了两种通过传统的字符注入方法和算法对抗机器学习（ML）规避技术绕过LLM提示注入和越狱检测系统的方法。通过对六个著名的保护系统进行测试，包括微软的Azure Prompt Shield和Meta的Prompt Guard，我们表明这两种方法都可以用来逃避检测，同时保持对抗效用，在某些情况下达到100%的逃避成功。此外，我们还证明，对手可以通过利用离线白盒模型计算的单词重要性排名来提高针对黑盒目标的攻击成功率（ASB）。我们的研究结果揭示了当前LLM保护机制中的漏洞，并强调了对更坚固的护栏系统的需求。



## **12. IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector**

iPad：人工智能检测的反向提示--一个强大且可解释的LLM生成文本检测器 cs.LG

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2502.15902v2) [paper-pdf](http://arxiv.org/pdf/2502.15902v2)

**Authors**: Zheng Chen, Yushi Feng, Changyang He, Yue Deng, Hongxi Pu, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinction between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution (OOD) data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.

摘要: 大型语言模型（LLM）在文本生成方面已经达到了人类水平的流畅性，这使得人类书写的文本和LLM生成的文本之间的区别变得复杂。这增加了误用的风险，并凸显了对可靠检测器的需求。然而，现有的检测器对非分布（OOD）数据和受攻击数据的鲁棒性较差，这对于现实世界场景至关重要。此外，他们很难提供可解释的证据来支持他们的决定，从而削弱了可靠性。鉴于这些挑战，我们提出了iPad（人工智能检测反向提示），这是一个新颖的框架，由一个提示反向器和两个区分器组成，用于识别可能生成输入文本的预测提示，用于检查输入文本与预测提示对齐的可能性。经验评估表明，iPad在分发内数据方面比最强基线高出9.05%（平均召回），在分发外（OOD）数据方面比最强基线高出12.93%（AUROC），在受攻击数据方面比最强基线高出5.48%（AUROC）。iPad还在结构化数据集上表现出色。此外，还进行了可解释性评估，以说明iPad通过允许用户直接检查决策证据来增强了人工智能检测的可信度，从而为其最先进的检测结果提供了可解释的支持。



## **13. The Man Behind the Sound: Demystifying Audio Private Attribute Profiling via Multimodal Large Language Model Agents**

声音背后的人：通过多模式大型语言模型代理揭开音频私人属性分析的神秘面纱 cs.CR

22 pages, 4 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10016v1) [paper-pdf](http://arxiv.org/pdf/2507.10016v1)

**Authors**: Lixu Wang, Kaixiang Yao, Xinfeng Li, Dong Yang, Haoyang Li, Xiaofeng Wang, Wei Dong

**Abstract**: Our research uncovers a novel privacy risk associated with multimodal large language models (MLLMs): the ability to infer sensitive personal attributes from audio data -- a technique we term audio private attribute profiling. This capability poses a significant threat, as audio can be covertly captured without direct interaction or visibility. Moreover, compared to images and text, audio carries unique characteristics, such as tone and pitch, which can be exploited for more detailed profiling. However, two key challenges exist in understanding MLLM-employed private attribute profiling from audio: (1) the lack of audio benchmark datasets with sensitive attribute annotations and (2) the limited ability of current MLLMs to infer such attributes directly from audio. To address these challenges, we introduce AP^2, an audio benchmark dataset that consists of two subsets collected and composed from real-world data, and both are annotated with sensitive attribute labels. Additionally, we propose Gifts, a hybrid multi-agent framework that leverages the complementary strengths of audio-language models (ALMs) and large language models (LLMs) to enhance inference capabilities. Gifts employs an LLM to guide the ALM in inferring sensitive attributes, then forensically analyzes and consolidates the ALM's inferences, overcoming severe hallucinations of existing ALMs in generating long-context responses. Our evaluations demonstrate that Gifts significantly outperforms baseline approaches in inferring sensitive attributes. Finally, we investigate model-level and data-level defense strategies to mitigate the risks of audio private attribute profiling. Our work validates the feasibility of audio-based privacy attacks using MLLMs, highlighting the need for robust defenses, and provides a dataset and framework to facilitate future research.

摘要: 我们的研究揭示了与多模式大型语言模型（MLLM）相关的新型隐私风险：从音频数据中推断敏感个人属性的能力--我们将这种技术称为音频私人属性剖析。这种能力构成了重大威胁，因为音频可以在没有直接交互或可见性的情况下被秘密捕获。此外，与图像和文本相比，音频具有独特的特征，例如音调和音调，可以利用这些特征进行更详细的分析。然而，在理解MLLM采用的音频私有属性分析方面存在两个关键挑战：（1）缺乏具有敏感属性注释的音频基准数据集;（2）当前MLLM直接从音频推断此类属性的能力有限。为了解决这些挑战，我们引入了AP ' 2，这是一个音频基准数据集，由从现实世界数据收集和组成的两个子集组成，并且两者都用敏感属性标签进行了注释。此外，我们还提出了Gifts，这是一种混合多智能体框架，利用音频语言模型（ILM）和大型语言模型（LLM）的互补优势来增强推理能力。Gifts使用LLM来指导ILM推断敏感属性，然后进行取证分析和巩固ILM的推论，克服现有ILM在生成长背景反应方面的严重幻觉。我们的评估表明，Gifts在推断敏感属性方面显着优于基线方法。最后，我们研究模型级和数据级防御策略，以降低音频私有属性分析的风险。我们的工作验证了使用MLLM进行基于音频的隐私攻击的可行性，强调了强大防御的必要性，并提供了一个数据集和框架来促进未来的研究。



## **14. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

并非所有令牌都是平等的：用于人工智能生成文本检测的困惑注意力加权网络 cs.CL

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2501.03940v3) [paper-pdf](http://arxiv.org/pdf/2501.03940v3)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.

摘要: 大型语言模型（LLM）的快速发展显着增强了它们生成连贯且上下文相关文本的能力，引发了人们对人工智能生成内容滥用的担忧，并使检测它变得至关重要。然而，这项任务仍然具有挑战性，特别是在未知的领域或不熟悉的LLM中。利用LLM下一个代币分发输出提供了一种理论上有吸引力的检测方法，因为它们包含了模型对不同数据库进行的广泛预训练的见解。尽管有希望，但试图实现这些输出的零射击方法收效有限。我们假设问题之一是，他们使用平均值来汇总各个代币之间的下一个代币分布指标，而有些代币自然更容易或更难预测，并且应该采用不同的加权方式。基于这个想法，我们提出了困惑注意力加权网络（PAWN），它使用LLM的最后隐藏状态和位置来根据整个序列长度的下一个令牌分布的指标对一系列特征的总和进行加权。尽管不是零射击，但我们的方法允许我们在磁盘上缓存最后一个隐藏状态和下一个令牌分布指标，从而大大减少了训练资源需求。PAWN在可训练参数仅为一小部分的情况下表现出比最强基线（微调LM）有竞争力甚至更好的分布性能。我们的模型还可以更好地推广到不可见的域和源模型，分布变化中决策边界的变异性较小。它对对抗攻击也更稳健，如果主干具有多语言能力，它会对监督训练期间未见过的语言进行良好的概括，LLaMA 3 -1B在与九种语言的交叉验证中达到了81.46%的平均宏平均F1得分。



## **15. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

EVALOOP：从自我一致性的角度评估LLM编程稳健性 cs.SE

20 pages, 11 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2505.12185v3) [paper-pdf](http://arxiv.org/pdf/2505.12185v3)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.

摘要: 评估大型语言模型（LLM）的编程能力对于它们在软件工程中的有效使用至关重要。然而，当前的评估主要衡量静态基准上生成的代码的准确性，忽视了编程任务期间模型稳健性的关键方面。虽然对抗性攻击提供了有关模型稳健性的见解，但它们的有效性有限，并且评估可能会受到限制。当前用于稳健性评估的对抗攻击方法会产生不一致的结果，难以在不同的LLM之间提供统一的评估。我们引入EVALOOP，这是一种新型评估框架，从自一致性的角度评估稳健性，即利用流行软件工程任务中固有的自然二重性，例如，代码生成和代码摘要。EVALOOP启动独立反馈循环：LLM生成输出（例如，代码）来自输入（例如，自然语言规范），然后使用生成的输出作为输入来产生新的输出（例如，将该代码总结为新规范）。EVALOOP重复该过程以评估每个循环中EVALOOP的有效性。这种循环策略本质上评估稳健性，而不依赖任何外部攻击设置，提供了一个统一的指标来评估LLM在编程中的稳健性。我们评估了16个著名的LLM（例如，GPT-4.1，O 4-mini）在EVALOOP上发现EVALOOP通常会在十个循环内导致pass@1性能绝对下降5.01%-19.31%。有趣的是，稳健性并不总是与初始性能一致（即，一次性查询）;例如，GPT-3.5-Turbo尽管初始代码生成优于DeepSeek-V2，但在重复评估循环中表现出较低的鲁棒性。



## **16. Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats**

博弈论与法学硕士和抽象人工智能相遇：为智能威胁时代重新构想网络安全 cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10621v1) [paper-pdf](http://arxiv.org/pdf/2507.10621v1)

**Authors**: Quanyan Zhu

**Abstract**: Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation.   The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset.   This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems.

摘要: 保护网络空间不仅需要先进的工具，还需要改变我们对威胁、信任和自主性的推理方式。传统的网络安全方法依赖于手动响应和脆弱的启发式方法。为了构建主动和智能的防御系统，我们需要集成的理论框架和软件工具。博弈论为对抗行为建模、设计战略防御和实现自治系统信任提供了严格的基础。与此同时，软件工具处理网络数据、可视化攻击表面、验证合规性并建议缓解措施。然而，理论和实际实施之间仍然存在脱节。   大型语言模型（LLM）和代理人工智能的兴起为弥合这一差距提供了一条新的途径。LLM支持的代理可以将抽象策略实施为现实世界的决策。相反，博弈论可以为这些代理在复杂工作流程中的推理和协调提供信息。LLM还挑战了经典的博弈论假设，例如完美理性或静态收益，促使新模型与认知和计算现实保持一致。这种协同进化提供了更丰富的理论基础和新的解决方案概念。人工智能还重塑了软件设计：系统现在必须从一开始就具有模块化、自适应性和信任意识。   本章探讨博弈论、代理人工智能和网络安全的交叉点。我们回顾了关键的博弈论框架（例如，静态、动态、Bayesian和Signal Game）和解决方案概念。然后，我们研究LLM代理如何增强网络防御并引入LLM驱动的将推理嵌入AI代理的游戏。最后，我们探索多代理工作流程和协调游戏，概述了这种融合如何培养安全、智能和自适应的网络系统。



## **17. LaSM: Layer-wise Scaling Mechanism for Defending Pop-up Attack on GUI Agents**

LaSM：用于防御对图形用户界面代理弹出攻击的分层扩展机制 cs.CR

10 pages, 9 figures

**SubmitDate**: 2025-07-13    [abs](http://arxiv.org/abs/2507.10610v1) [paper-pdf](http://arxiv.org/pdf/2507.10610v1)

**Authors**: Zihe Yan, Zhuosheng Zhang

**Abstract**: Graphical user interface (GUI) agents built on multimodal large language models (MLLMs) have recently demonstrated strong decision-making abilities in screen-based interaction tasks. However, they remain highly vulnerable to pop-up-based environmental injection attacks, where malicious visual elements divert model attention and lead to unsafe or incorrect actions. Existing defense methods either require costly retraining or perform poorly under inductive interference. In this work, we systematically study how such attacks alter the attention behavior of GUI agents and uncover a layer-wise attention divergence pattern between correct and incorrect outputs. Based on this insight, we propose \textbf{LaSM}, a \textit{Layer-wise Scaling Mechanism} that selectively amplifies attention and MLP modules in critical layers. LaSM improves the alignment between model saliency and task-relevant regions without additional training. Extensive experiments across 12 types of pop-up perturbations and 4 different model backbones show that LaSM consistently enhances the defense success rate. When combined with prompt-level alerts, LaSM achieves over 98\% robustness even under strong inductive attacks. Our findings reveal that attention misalignment is a core vulnerability in MLLM agents and can be effectively addressed through selective layer-wise modulation.

摘要: 建立在多模式大型语言模型（MLLM）上的图形用户界面（图形用户界面）代理最近在基于屏幕的交互任务中表现出了强大的决策能力。然而，它们仍然非常容易受到基于弹出窗口的环境注入攻击，恶意视觉元素会转移模型的注意力并导致不安全或不正确的操作。现有的防御方法要么需要昂贵的再培训，要么在感应干扰下表现不佳。在这项工作中，我们系统地研究此类攻击如何改变图形用户界面代理的注意力行为，并揭示正确和不正确输出之间的分层注意力分歧模式。基于这一见解，我们提出了\textBF{LaSM}，这是一种\textit{逐层缩放机制}，可以选择性地放大关键层中的注意力和MLP模块。LaSM无需额外培训即可改善模型显着性和任务相关区域之间的一致性。针对12种弹出扰动和4种不同模型主干的广泛实验表明，LaSM持续提高了防御成功率。与预算级警报相结合时，即使在强诱导攻击下，LaSM也能实现超过98%的稳健性。我们的研究结果表明，注意力错位是MLLM代理的核心漏洞，可以通过选择性分层调制有效地解决。



## **18. Auditing Prompt Caching in Language Model APIs**

语言模型API中的审核提示缓存 cs.CL

Accepted at ICML 2025

**SubmitDate**: 2025-07-13    [abs](http://arxiv.org/abs/2502.07776v2) [paper-pdf](http://arxiv.org/pdf/2502.07776v2)

**Authors**: Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto

**Abstract**: Prompt caching in large language models (LLMs) results in data-dependent timing variations: cached prompts are processed faster than non-cached prompts. These timing differences introduce the risk of side-channel timing attacks. For example, if the cache is shared across users, an attacker could identify cached prompts from fast API response times to learn information about other users' prompts. Because prompt caching may cause privacy leakage, transparency around the caching policies of API providers is important. To this end, we develop and conduct statistical audits to detect prompt caching in real-world LLM API providers. We detect global cache sharing across users in seven API providers, including OpenAI, resulting in potential privacy leakage about users' prompts. Timing variations due to prompt caching can also result in leakage of information about model architecture. Namely, we find evidence that OpenAI's embedding model is a decoder-only Transformer, which was previously not publicly known.

摘要: 大型语言模型（LLM）中的提示缓存会导致依赖于数据的时间变化：缓存的提示比非缓存的提示处理得更快。这些定时差异引入了侧信道定时攻击的风险。例如，如果缓存在用户之间共享，攻击者可以从快速API响应时间中识别缓存的提示，以了解有关其他用户提示的信息。由于即时缓存可能会导致隐私泄露，因此API提供商缓存策略的透明度非常重要。为此，我们开发并进行统计审计，以检测现实世界LLM API提供商中的即时缓存。我们在包括OpenAI在内的七个API提供商中检测到用户之间的全局缓存共享，从而导致用户提示的潜在隐私泄露。由于提示缓存而导致的时间变化也可能导致有关模型架构的信息泄露。也就是说，我们发现证据表明OpenAI的嵌入模型是一个仅解码器的Transformer，这在以前并不为人所知。



## **19. LLMalMorph: On The Feasibility of Generating Variant Malware using Large-Language-Models**

LLMalMorph：关于使用大型语言模型生成变体恶意软件的可行性 cs.CR

13 pages

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2507.09411v1) [paper-pdf](http://arxiv.org/pdf/2507.09411v1)

**Authors**: Md Ajwad Akil, Adrian Shuai Li, Imtiaz Karim, Arun Iyengar, Ashish Kundu, Vinny Parla, Elisa Bertino

**Abstract**: Large Language Models (LLMs) have transformed software development and automated code generation. Motivated by these advancements, this paper explores the feasibility of LLMs in modifying malware source code to generate variants. We introduce LLMalMorph, a semi-automated framework that leverages semantical and syntactical code comprehension by LLMs to generate new malware variants. LLMalMorph extracts function-level information from the malware source code and employs custom-engineered prompts coupled with strategically defined code transformations to guide the LLM in generating variants without resource-intensive fine-tuning. To evaluate LLMalMorph, we collected 10 diverse Windows malware samples of varying types, complexity and functionality and generated 618 variants. Our thorough experiments demonstrate that it is possible to reduce the detection rates of antivirus engines of these malware variants to some extent while preserving malware functionalities. In addition, despite not optimizing against any Machine Learning (ML)-based malware detectors, several variants also achieved notable attack success rates against an ML-based malware classifier. We also discuss the limitations of current LLM capabilities in generating malware variants from source code and assess where this emerging technology stands in the broader context of malware variant generation.

摘要: 大型语言模型（LLM）改变了软件开发和自动化代码生成。受这些进步的启发，本文探讨了LLM修改恶意软件源代码以生成变体的可行性。我们引入LLMalMorph，这是一个半自动化框架，利用LLM的语义和语法代码理解来生成新的恶意软件变体。LLMalMorph从恶意软件源代码中提取功能级信息，并采用定制工程提示与战略定义的代码转换相结合，以指导LLM生成变体，而无需进行资源密集型微调。为了评估LLMalMorph，我们收集了10个类型、复杂性和功能不同的Windows恶意软件样本，并生成了618个变体。我们彻底的实验表明，可以在一定程度上降低这些恶意软件变体的防病毒引擎的检测率，同时保留恶意软件功能。此外，尽管没有针对任何基于机器学习（ML）的恶意软件检测器进行优化，但几个变体也针对基于ML的恶意软件分类器实现了显着的攻击成功率。我们还讨论了当前LLM功能在从源代码生成恶意软件变体方面的局限性，并评估这项新兴技术在恶意软件变体生成的更广泛背景下的地位。



## **20. Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers**

对抗激活修补：检测和减轻安全调整变形金刚中紧急欺骗的框架 cs.LG

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2507.09406v1) [paper-pdf](http://arxiv.org/pdf/2507.09406v1)

**Authors**: Santhosh Kumar Ravindran

**Abstract**: Large language models (LLMs) aligned for safety through techniques like reinforcement learning from human feedback (RLHF) often exhibit emergent deceptive behaviors, where outputs appear compliant but subtly mislead or omit critical information. This paper introduces adversarial activation patching, a novel mechanistic interpretability framework that leverages activation patching as an adversarial tool to induce, detect, and mitigate such deception in transformer-based models. By sourcing activations from "deceptive" prompts and patching them into safe forward passes at specific layers, we simulate vulnerabilities and quantify deception rates. Through toy neural network simulations across multiple scenarios (e.g., 1000 trials per setup), we demonstrate that adversarial patching increases deceptive outputs to 23.9% from a 0% baseline, with layer-specific variations supporting our hypotheses. We propose six hypotheses, including transferability across models, exacerbation in multimodal settings, and scaling effects. An expanded literature review synthesizes over 20 key works in interpretability, deception, and adversarial attacks. Mitigation strategies, such as activation anomaly detection and robust fine-tuning, are detailed, alongside ethical considerations and future research directions. This work advances AI safety by highlighting patching's dual-use potential and provides a roadmap for empirical studies on large-scale models.

摘要: 通过人类反馈强化学习（RL HF）等技术实现安全性调整的大型语言模型（LLM）通常表现出紧急欺骗行为，其中输出看起来合规，但微妙地误导或省略关键信息。本文介绍了对抗性激活补丁，这是一种新型的机械解释性框架，它利用激活补丁作为对抗性工具来诱导、检测和减轻基于转换器的模型中的此类欺骗。通过从“欺骗性”提示中获取激活并将其修补为特定层的安全转发传递，我们模拟漏洞并量化欺骗率。通过跨多个场景的玩具神经网络模拟（例如，每个设置1000次试验），我们证明对抗性修补将欺骗性输出从0%基线增加到23.9%，特定层的变化支持我们的假设。我们提出了六个假设，包括模型之间的可移植性、多模式环境的恶化以及缩放效应。扩大的文献评论综合了可解释性、欺骗性和对抗性攻击方面的20多部关键作品。详细介绍了缓解策略，例如激活异常检测和稳健的微调，以及道德考虑和未来的研究方向。这项工作通过强调修补的双重用途潜力来提高人工智能安全性，并为大规模模型的实证研究提供了路线图。



## **21. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2401.17256v4) [paper-pdf](http://arxiv.org/pdf/2401.17256v4)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **22. A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1**

基于LLM的论点分类的全面研究：从LLAMA到GPT-4 o到Deepseek-R1 cs.CL

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08621v1) [paper-pdf](http://arxiv.org/pdf/2507.08621v1)

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.

摘要: 论据挖掘（AM）是一个跨学科研究领域，集成了逻辑、哲学、语言学、修辞学、法学、心理学和计算机科学的见解。它涉及自动识别和提取论点成分（例如前提和主张），以及检测它们之间的关系（例如支持、攻击或中立）。最近，该领域取得了显着的进步，特别是随着大型语言模型（LLM）的出现，与传统方法和其他深度学习模型相比，LLM提高了分析和提取参数语义的效率。有许多用于测试和验证LLM质量的基准，但在公开可用的论点分类数据库中仍然缺乏有关这些模型操作的研究和结果。本文使用Args.me和UKP等不同数据集对LLM进行了一项研究。测试的模型包括GPT、Llama和DeepSeek的版本，以及包含思想链算法的推理增强变体。结果表明，ChatGPT-4 o在论点分类基准方面优于其他。在包含推理能力的模型中，Deepseek-R1显示出其优越性。然而，尽管具有优势，GPT-4 o和Deepseek-R1仍然会犯错误。讨论了所有模型最常见的错误。据我们所知，所介绍的工作是首次使用LLM和提示算法对上述数据集进行更广泛的分析。该工作还揭示了已知提示算法在论据分析中的一些弱点，同时指出了改进的方向。这项工作的附加值是对可用论点数据集的深入分析并展示其缺点。



## **23. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

LLM基于代理的完全计算机接管攻击的阴暗面 cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.06850v3) [paper-pdf](http://arxiv.org/pdf/2507.06850v3)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.

摘要: 大型语言模型（LLM）代理和多代理系统的快速采用使自然语言处理和生成具有前所未有的能力。然而，这些系统引入了前所未有的安全漏洞，超出了传统的即时注入攻击的范围。本文首次对LLM代理进行了全面评估，作为攻击载体，这些攻击载体能够通过利用自主实体相互交互和影响的代理人工智能系统内的信任边界来实现完全的计算机接管。我们证明，对手可以利用三种不同的攻击表面--直接提示注入、RAG后门攻击和代理间信任利用--来强迫流行的LLM（包括GPT-4 o、Claude-4和Gemini-2.5）在受害者机器上自主安装和执行恶意软件。我们对17个最先进的LLM的评估揭示了一个令人震惊的漏洞层次结构：虽然41.2%的模型屈服于直接即时注入，但52.9%的模型容易受到RAG后门攻击，并且关键的82.4%可以通过代理间信任利用而受到损害。值得注意的是，我们发现成功抵抗直接恶意命令的LLM将在对等代理请求时执行相同的有效负载，这揭示了当前多代理安全模型中的一个根本缺陷。我们的研究结果表明，只有5.9%的测试模型（1/17）被证明能够抵抗所有攻击载体，其中大多数表现出依赖于上下文的安全行为，从而创建了可利用的盲点。我们的研究结果还强调了提高对LLM安全风险的认识和研究的必要性，这表明网络安全威胁的范式转变，人工智能工具本身成为复杂的攻击载体。



## **24. Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection**

Emoji攻击：增强针对LLM法官检测的越狱攻击 cs.CL

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2411.01077v4) [paper-pdf](http://arxiv.org/pdf/2411.01077v4)

**Authors**: Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson

**Abstract**: Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.

摘要: 越狱技术欺骗大型语言模型（LLM）产生受限输出，构成潜在威胁。一种防御措施是使用另一位LLM作为法官来评估生成文本的危害性。然而，我们发现这些Judge LLM很容易受到标记分割偏见的影响，当分隔符改变标记化过程、将单词分割成更小的子标记时，就会出现这个问题。这会改变整个序列的嵌入，降低检测准确性，并允许有害内容被错误分类为安全内容。在本文中，我们介绍了Emoji Attack，这是一种新颖的策略，通过利用代币分割偏见来放大现有的越狱提示。我们的方法利用上下文学习，在LLM法官评估文本之前系统地将表情符号插入文本中，从而引发嵌入失真，从而显着降低检测到不安全内容的可能性。与传统的分隔符不同，表情符号还会引入语义歧义，使它们在这种攻击中特别有效。通过对最先进的Judge LLM的实验，我们证明Emoji Attack大幅降低了不安全的预测率，绕过了现有的保障措施。



## **25. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

从心理测量学角度通过攻击来评估大型语言模型中的内隐偏差 cs.CL

Accepted to ACL 2025 Findings

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2406.14023v5) [paper-pdf](http://arxiv.org/pdf/2406.14023v5)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.

摘要: 随着大型语言模型（LLM）成为信息获取的重要方式，人们越来越担心LLM可能会加剧不道德内容的传播，包括在没有明确有害词语的情况下伤害某些人群的隐性偏见。在本文中，我们进行了严格的评估LLM的隐性偏见对某些人口统计数据的攻击，从心理测量学的角度，以引起有偏见的观点的协议。受认知和社会心理学中心理测量原则的启发，我们提出了三种攻击方法，即伪装、欺骗和教导。综合相应的攻击指令，我们构建了两个基准：（1）双语数据集，其中包含涵盖四种偏见类型（2.7 K实例）的偏见陈述，用于广泛的比较分析，和（2）BUMBLE，一个跨越九种常见偏见类型（12.7 K实例）的更大基准，用于全面评估。对流行的商业和开源LLM的广泛评估表明，我们的方法比竞争基线更有效地引发LLM的内部偏见。我们的攻击方法和基准提供了评估LLM道德风险的有效手段，推动LLM在开发过程中实现更强的问责制。我们的代码、数据和基准可在https://yuchenwen1.github.io/ImplicitBiasEvaluation/上获取。



## **26. Invariant-based Robust Weights Watermark for Large Language Models**

大型语言模型的基于不变的鲁棒权重水印 cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08288v1) [paper-pdf](http://arxiv.org/pdf/2507.08288v1)

**Authors**: Qingxiao Guo, Xinjie Zhu, Yilong Ma, Hui Jin, Yunhao Wang, Weifeng Zhang, Xiaobing Guo

**Abstract**: Watermarking technology has gained significant attention due to the increasing importance of intellectual property (IP) rights, particularly with the growing deployment of large language models (LLMs) on billions resource-constrained edge devices. To counter the potential threats of IP theft by malicious users, this paper introduces a robust watermarking scheme without retraining or fine-tuning for transformer models. The scheme generates a unique key for each user and derives a stable watermark value by solving linear constraints constructed from model invariants. Moreover, this technology utilizes noise mechanism to hide watermark locations in multi-user scenarios against collusion attack. This paper evaluates the approach on three popular models (Llama3, Phi3, Gemma), and the experimental results confirm the strong robustness across a range of attack methods (fine-tuning, pruning, quantization, permutation, scaling, reversible matrix and collusion attacks).

摘要: 由于知识产权（IP）权的重要性日益增加，特别是随着大型语言模型（LLM）在数十亿个资源有限的边缘设备上部署的日益增多，水印技术受到了广泛关注。为了应对恶意用户IP盗窃的潜在威胁，本文引入了一种鲁棒的水印方案，无需对Transformer模型进行再培训或微调。该方案为每个用户生成唯一的密钥，并通过求解由模型不变量构建的线性约束来推导稳定的水印值。此外，该技术利用噪音机制来隐藏多用户场景中的水印位置，以防止共谋攻击。本文在三种流行模型（Llama 3、Phi 3、Gemma）上评估了该方法，实验结果证实了一系列攻击方法（微调、修剪、量化、置换、缩放、可逆矩阵和共谋攻击）的强大鲁棒性。



## **27. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

突破安全极限：2025年ATLAS挑战赛技术报告 cs.CR

AdvML@CVPR Challenge Report

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2506.12430v2) [paper-pdf](http://arxiv.org/pdf/2506.12430v2)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.

摘要: 多模式大型语言模型（MLLM）在不同的应用程序中实现了变革性的进步，但仍然容易受到安全威胁，尤其是引发有害输出的越狱攻击。为了系统地评估和提高其安全性，我们组织了对抗性测试和大模型对齐安全大挑战赛（ATLAS）2025。本技术报告介绍了比赛的结果，其中86个团队通过对抗性图像文本攻击分两个阶段测试MLLM漏洞：白盒和黑匣子评估。竞赛结果凸显了确保MLLM方面持续存在的挑战，并为开发更强大的防御机制提供了宝贵的指导。该挑战为MLLM安全评估建立了新的基准，并为推进更安全的多模式人工智能系统奠定了基础。此挑战的代码和数据可在https://github.com/NY1024/ATLAS_Challenge_2025上公开获取。



## **28. A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking**

动态Stackelberg游戏框架，用于针对LLM越狱的大型人工智能防御 cs.AI

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08207v1) [paper-pdf](http://arxiv.org/pdf/2507.08207v1)

**Authors**: Zhengye Han, Quanyan Zhu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking.

摘要: 随着大型语言模型（LLM）越来越多地部署在关键应用程序中，越狱的挑战（对手操纵模型以绕过安全机制）已成为一个重大问题。本文提出了一个动态Stackelberg博弈框架，来建模LLM越狱背景下攻击者和防御者之间的互动。该框架将预算-响应动态视为一个顺序扩展形式的游戏，其中防御者作为领导者，承诺采取策略，同时预测攻击者的最佳响应。我们提出了一种新型的代理人工智能解决方案，即“紫色代理”，它使用快速探索随机树（RTI）集成了对抗性探索和防御策略。Purple Agent主动模拟潜在的攻击轨迹，并主动干预以防止有害输出。这种方法提供了一种分析对抗动态的原则性方法，并为减轻越狱风险提供了基础。



## **29. Beyond the Worst Case: Extending Differential Privacy Guarantees to Realistic Adversaries**

超越最坏情况：将差异隐私保证扩展到现实对手 cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08158v1) [paper-pdf](http://arxiv.org/pdf/2507.08158v1)

**Authors**: Marika Swanberg, Meenatchi Sundaram Muthu Selva Annamalai, Jamie Hayes, Borja Balle, Adam Smith

**Abstract**: Differential Privacy (DP) is a family of definitions that bound the worst-case privacy leakage of a mechanism. One important feature of the worst-case DP guarantee is it naturally implies protections against adversaries with less prior information, more sophisticated attack goals, and complex measures of a successful attack. However, the analytical tradeoffs between the adversarial model and the privacy protections conferred by DP are not well understood thus far. To that end, this work sheds light on what the worst-case guarantee of DP implies about the success of attackers that are more representative of real-world privacy risks.   In this paper, we present a single flexible framework that generalizes and extends the patchwork of bounds on DP mechanisms found in prior work. Our framework allows us to compute high-probability guarantees for DP mechanisms on a large family of natural attack settings that previous bounds do not capture. One class of such settings is the approximate reconstruction of multiple individuals' data, such as inferring nearly entire columns of a tabular data set from noisy marginals and extracting sensitive information from DP-trained language models.   We conduct two empirical case studies to illustrate the versatility of our bounds and compare them to the success of state-of-the-art attacks. Specifically, we study attacks that extract non-uniform PII from a DP-trained language model, as well as multi-column reconstruction attacks where the adversary has access to some columns in the clear and attempts to reconstruct the remaining columns for each person's record. We find that the absolute privacy risk of attacking non-uniform data is highly dependent on the adversary's prior probability of success. Our high probability bounds give us a nuanced understanding of the privacy leakage of DP mechanisms in a variety of previously understudied attack settings.

摘要: 差异隐私（DP）是一系列定义，限制了机制的最坏情况隐私泄露。最坏情况DP保证的一个重要特征是，它自然意味着针对先验信息较少、攻击目标更复杂且成功攻击措施复杂的对手提供保护。然而，迄今为止，对抗模型和DP赋予的隐私保护之间的分析权衡还没有得到很好的理解。为此，这项工作揭示了DP的最坏情况保证对更能代表现实世界隐私风险的攻击者的成功意味着什么。   在本文中，我们提出了一个灵活的框架，该框架概括和扩展了先前工作中发现的DP机制边界的拼凑。我们的框架允许我们在以前的界限无法捕捉的一大系列自然攻击设置上计算DP机制的高概率保证。一类此类设置是多个人数据的大致重建，例如从有噪的边缘推断表格数据集的几乎整个列，并从DP训练的语言模型中提取敏感信息。   我们进行了两个实证案例研究，以说明我们边界的多功能性，并将它们与最先进的攻击的成功进行比较。具体来说，我们研究了从DP训练的语言模型中提取非均匀PRI的攻击，以及多列重建攻击，其中对手可以以明文方式访问某些列并试图为每个人的记录重建剩余列。我们发现，攻击非均匀数据的绝对隐私风险高度取决于对手的先验成功概率。我们的高概率界限让我们对各种以前未充分研究的攻击环境中DP机制的隐私泄露有了细致入微的了解。



## **30. Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)**

为Red-Teaming大型语言模型（LLM）操作威胁模型 cs.CL

Transactions of Machine Learning Research (TMLR)

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2407.14937v2) [paper-pdf](http://arxiv.org/pdf/2407.14937v2)

**Authors**: Apurv Verma, Satyapriya Krishna, Sebastian Gehrmann, Madhavan Seshadri, Anu Pradhan, Tom Ault, Leslie Barrett, David Rabinowitz, John Doucette, NhatHai Phan

**Abstract**: Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.

摘要: 使用大型语言模型（LLM）创建安全且有弹性的应用程序需要预测、调整和应对不可预见的威胁。红色团队已成为识别现实世界LLM实施中漏洞的关键技术。本文提出了一个详细的威胁模型，并提供了对LLM的红色团队攻击的知识系统化（SoK）。我们根据LLM开发和部署过程的阶段开发攻击分类，并从之前的研究中提取各种见解。此外，我们还为从业者编写了防御方法和实用的红色团队策略。通过描述突出的攻击主题并揭示各种切入点，本文提供了一个框架来提高基于LLM的系统的安全性和稳健性。



## **31. Defending Against Prompt Injection With a Few DefensiveTokens**

使用一些防御代币来防御即时注射 cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07974v1) [paper-pdf](http://arxiv.org/pdf/2507.07974v1)

**Authors**: Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, David Wagner

**Abstract**: When large language model (LLM) systems interact with external data to perform complex tasks, a new attack, namely prompt injection, becomes a significant threat. By injecting instructions into the data accessed by the system, the attacker is able to override the initial user task with an arbitrary task directed by the attacker. To secure the system, test-time defenses, e.g., defensive prompting, have been proposed for system developers to attain security only when needed in a flexible manner. However, they are much less effective than training-time defenses that change the model parameters. Motivated by this, we propose DefensiveToken, a test-time defense with prompt injection robustness comparable to training-time alternatives. DefensiveTokens are newly inserted as special tokens, whose embeddings are optimized for security. In security-sensitive cases, system developers can append a few DefensiveTokens before the LLM input to achieve security with a minimal utility drop. In scenarios where security is less of a concern, developers can simply skip DefensiveTokens; the LLM system remains the same as there is no defense, generating high-quality responses. Thus, DefensiveTokens, if released alongside the model, allow a flexible switch between the state-of-the-art (SOTA) utility and almost-SOTA security at test time. The code is available at https://github.com/Sizhe-Chen/DefensiveToken.

摘要: 当大型语言模型（LLM）系统与外部数据交互以执行复杂任务时，一种新的攻击（即提示注入）将成为重大威胁。通过将指令注入系统访问的数据中，攻击者能够用攻击者指示的任意任务覆盖初始用户任务。为了保护系统，测试时防御措施，例如防御性提示已被建议供系统开发人员仅在需要时以灵活的方式获得安全性。然而，它们比改变模型参数的训练时防御有效得多。出于此动机，我们提出了DefensiveToken，这是一种测试时防御，具有与训练时替代方案相当的即时注入鲁棒性。DefensiveTokens作为特殊令牌新插入，其嵌入针对安全性进行了优化。在安全敏感的情况下，系统开发人员可以在LLM输入之前添加一些DefensiveTokens，以最小的实用程序下降来实现安全性。在安全性不太值得关注的场景中，开发人员可以简单地跳过DefensiveTokens; LLM系统由于没有防御而保持不变，从而生成高质量的响应。因此，DefensiveTokens如果与该模型一起发布，将允许在测试时在最先进的（SOTA）实用程序和几乎SOTA安全性之间灵活切换。该代码可在https://github.com/Sizhe-Chen/DefensiveToken上获取。



## **32. Evaluating Robustness of Large Audio Language Models to Audio Injection: An Empirical Study**

评估大型音频语言模型对音频注入的稳健性：一项实证研究 cs.CL

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2505.19598v2) [paper-pdf](http://arxiv.org/pdf/2505.19598v2)

**Authors**: Guanyu Hou, Jiaming He, Yinhang Zhou, Ji Guo, Yitong Qiao, Rui Zhang, Wenbo Jiang

**Abstract**: Large Audio-Language Models (LALMs) are increasingly deployed in real-world applications, yet their robustness against malicious audio injection attacks remains underexplored. This study systematically evaluates five leading LALMs across four attack scenarios: Audio Interference Attack, Instruction Following Attack, Context Injection Attack, and Judgment Hijacking Attack. Using metrics like Defense Success Rate, Context Robustness Score, and Judgment Robustness Index, their vulnerabilities and resilience were quantitatively assessed. Experimental results reveal significant performance disparities among models; no single model consistently outperforms others across all attack types. The position of malicious content critically influences attack effectiveness, particularly when placed at the beginning of sequences. A negative correlation between instruction-following capability and robustness suggests models adhering strictly to instructions may be more susceptible, contrasting with greater resistance by safety-aligned models. Additionally, system prompts show mixed effectiveness, indicating the need for tailored strategies. This work introduces a benchmark framework and highlights the importance of integrating robustness into training pipelines. Findings emphasize developing multi-modal defenses and architectural designs that decouple capability from susceptibility for secure LALMs deployment.

摘要: 大型音频语言模型（LALM）越来越多地部署在现实世界的应用程序中，但它们针对恶意音频注入攻击的稳健性仍然没有得到充分的研究。本研究系统地评估了针对四种攻击场景的五种主要LALM：音频干扰攻击、指令跟随攻击、上下文注入攻击和判断劫持攻击。使用防御成功率、上下文稳健性得分和判断稳健性指数等指标，量化评估了他们的脆弱性和弹性。实验结果揭示了模型之间的显着性能差异;没有一个模型在所有攻击类型中始终优于其他模型。恶意内容的位置严重影响攻击的有效性，特别是当被放置在序列的开头时。指令遵循能力和稳健性之间的负相关性表明，严格遵守指令的模型可能更容易受到影响，而安全一致的模型则具有更大的抵抗力。此外，系统提示显示出好坏参半的有效性，表明需要定制策略。这项工作引入了基准框架，并强调了将稳健性集成到训练管道中的重要性。研究结果强调开发多模式防御和架构设计，将能力与安全LALM部署的敏感性脱钩。



## **33. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

“我很坏”：在音频语言模型中解释秘密、普遍和稳健的音频越狱 cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2502.00718v2) [paper-pdf](http://arxiv.org/pdf/2502.00718v2)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.

摘要: 多模式大型语言模型的兴起引入了创新的人机交互范式，但也给机器学习安全带来了重大挑战。由于口语交流的直观性，音频语言模型（ILM）尤其重要，但人们对其失败模式知之甚少。本文探讨了针对ILM的音频越狱，重点关注它们绕过对齐机制的能力。我们构建了跨越提示、任务甚至基本音频样本的对抗性扰动，展示了音频模式中的第一次普遍越狱，并表明这些在模拟的现实世界条件下仍然有效。除了证明攻击可行性之外，我们还分析了ILM如何解释这些音频对抗示例，并揭示它们来编码难以察觉的第一人称有毒语音-这表明用于引发有毒输出的最有效的干扰专门嵌入了音频信号中的语言特征。这些结果对于理解多模式模型中不同模式之间的相互作用具有重要意义，并为增强对抗性音频攻击的防御提供了可行的见解。



## **34. GuardVal: Dynamic Large Language Model Jailbreak Evaluation for Comprehensive Safety Testing**

GuardVal：用于全面安全测试的动态大语言模型越狱评估 cs.LG

24 pages

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07735v1) [paper-pdf](http://arxiv.org/pdf/2507.07735v1)

**Authors**: Peiyan Zhang, Haibo Jin, Liying Kang, Haohan Wang

**Abstract**: Jailbreak attacks reveal critical vulnerabilities in Large Language Models (LLMs) by causing them to generate harmful or unethical content. Evaluating these threats is particularly challenging due to the evolving nature of LLMs and the sophistication required in effectively probing their vulnerabilities. Current benchmarks and evaluation methods struggle to fully address these challenges, leaving gaps in the assessment of LLM vulnerabilities. In this paper, we review existing jailbreak evaluation practices and identify three assumed desiderata for an effective jailbreak evaluation protocol. To address these challenges, we introduce GuardVal, a new evaluation protocol that dynamically generates and refines jailbreak prompts based on the defender LLM's state, providing a more accurate assessment of defender LLMs' capacity to handle safety-critical situations. Moreover, we propose a new optimization method that prevents stagnation during prompt refinement, ensuring the generation of increasingly effective jailbreak prompts that expose deeper weaknesses in the defender LLMs. We apply this protocol to a diverse set of models, from Mistral-7b to GPT-4, across 10 safety domains. Our findings highlight distinct behavioral patterns among the models, offering a comprehensive view of their robustness. Furthermore, our evaluation process deepens the understanding of LLM behavior, leading to insights that can inform future research and drive the development of more secure models.

摘要: 越狱攻击揭示了大型语言模型（LLM）中的关键漏洞，导致它们生成有害或不道德的内容。评估这些威胁是特别具有挑战性的，由于不断变化的性质LLM和复杂性需要有效地探测其漏洞。目前的基准和评估方法难以充分应对这些挑战，在评估LLM脆弱性方面留下了空白。在本文中，我们回顾了现有的越狱评估实践，并确定了三个假设的必要条件，一个有效的越狱评估协议。为了应对这些挑战，我们引入了GuardVal，这是一种新的评估协议，可以根据防守方LLM的状态动态生成和改进越狱提示，从而对防守方LLM处理安全关键情况的能力提供更准确的评估。此外，我们提出了一种新的优化方法，可以防止在即时改进期间出现停滞，确保生成越来越有效的越狱提示，从而暴露防御者LLM中更深层次的弱点。我们将该协议应用于10个安全领域的一系列不同模型，从Mistral-7 b到GPT-4。我们的研究结果强调了模型之间不同的行为模式，并全面了解其稳健性。此外，我们的评估过程加深了对LLM行为的理解，从而获得了可以为未来研究提供信息并推动更安全模型的开发的见解。



## **35. May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks**

请注意吗？使用架构感知攻击突破基于微调的提示注入防御 cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07417v1) [paper-pdf](http://arxiv.org/pdf/2507.07417v1)

**Authors**: Nishit V. Pandya, Andrey Labunets, Sicun Gao, Earlence Fernandes

**Abstract**: A popular class of defenses against prompt injection attacks on large language models (LLMs) relies on fine-tuning the model to separate instructions and data, so that the LLM does not follow instructions that might be present with data. There are several academic systems and production-level implementations of this idea. We evaluate the robustness of this class of prompt injection defenses in the whitebox setting by constructing strong optimization-based attacks and showing that the defenses do not provide the claimed security properties. Specifically, we construct a novel attention-based attack algorithm for text-based LLMs and apply it to two recent whitebox defenses SecAlign (CCS 2025) and StruQ (USENIX Security 2025), showing attacks with success rates of up to 70% with modest increase in attacker budget in terms of tokens. Our findings make fundamental progress towards understanding the robustness of prompt injection defenses in the whitebox setting. We release our code and attacks at https://github.com/nishitvp/better_opts_attacks

摘要: 针对大型语言模型（LLM）的即时注入攻击的一类流行防御依赖于对模型进行微调以分离指令和数据，以便LLM不会遵循可能存在于数据中的指令。这个想法有几个学术系统和生产级实现。我们通过构建强大的基于优化的攻击并表明这些防御不提供声称的安全属性来评估白盒设置中此类即时注入防御的稳健性。具体来说，我们为基于文本的LLM构建了一种新颖的基于注意力的攻击算法，并将其应用于最近的两种白盒防御SecAlign（CCCS 2025）和StruQ（USENIX Security 2025），显示攻击成功率高达70%，攻击者预算在代币方面略有增加。我们的研究结果在理解白盒环境中即时注射防御的稳健性方面取得了根本性进展。我们在https://github.com/nishitvp/better_opts_attacks上发布我们的代码和攻击



## **36. Hybrid LLM-Enhanced Intrusion Detection for Zero-Day Threats in IoT Networks**

针对物联网网络中零日威胁的混合LLM增强型入侵检测 cs.CR

6 pages, IEEE conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07413v1) [paper-pdf](http://arxiv.org/pdf/2507.07413v1)

**Authors**: Mohammad F. Al-Hammouri, Yazan Otoum, Rasha Atwa, Amiya Nayak

**Abstract**: This paper presents a novel approach to intrusion detection by integrating traditional signature-based methods with the contextual understanding capabilities of the GPT-2 Large Language Model (LLM). As cyber threats become increasingly sophisticated, particularly in distributed, heterogeneous, and resource-constrained environments such as those enabled by the Internet of Things (IoT), the need for dynamic and adaptive Intrusion Detection Systems (IDSs) becomes increasingly urgent. While traditional methods remain effective for detecting known threats, they often fail to recognize new and evolving attack patterns. In contrast, GPT-2 excels at processing unstructured data and identifying complex semantic relationships, making it well-suited to uncovering subtle, zero-day attack vectors. We propose a hybrid IDS framework that merges the robustness of signature-based techniques with the adaptability of GPT-2-driven semantic analysis. Experimental evaluations on a representative intrusion dataset demonstrate that our model enhances detection accuracy by 6.3%, reduces false positives by 9.0%, and maintains near real-time responsiveness. These results affirm the potential of language model integration to build intelligent, scalable, and resilient cybersecurity defences suited for modern connected environments.

摘要: 本文通过将传统的基于签名的方法与GPT-2大型语言模型（LLM）的上下文理解能力集成，提出了一种新颖的入侵检测方法。随着网络威胁变得越来越复杂，特别是在分布式、异类和资源受限的环境中，例如物联网（IoT）所支持的环境中，对动态和自适应入侵检测系统（IDS）的需求变得越来越紧迫。虽然传统方法对于检测已知威胁仍然有效，但它们通常无法识别新的和不断发展的攻击模式。相比之下，GPT-2擅长处理非结构化数据和识别复杂的语义关系，因此非常适合发现微妙的零日攻击载体。我们提出了一个混合IDS框架，该框架将基于签名的技术的稳健性与GPT-2驱动的语义分析的适应性相结合。对代表性入侵数据集的实验评估表明，我们的模型将检测准确性提高了6.3%，将假阳性降低了9.0%，并保持了近乎实时的响应能力。这些结果证实了语言模型集成在构建适合现代互联环境的智能、可扩展和弹性网络安全防御方面的潜力。



## **37. Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models**

Gen-AI时代的网络钓鱼检测：量化LLM与经典模型 cs.CR

8 Pages, IEEE Conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07406v1) [paper-pdf](http://arxiv.org/pdf/2507.07406v1)

**Authors**: Jikesh Thapa, Gurrehmat Chahal, Serban Voinea Gabreanu, Yazan Otoum

**Abstract**: Phishing attacks are becoming increasingly sophisticated, underscoring the need for detection systems that strike a balance between high accuracy and computational efficiency. This paper presents a comparative evaluation of traditional Machine Learning (ML), Deep Learning (DL), and quantized small-parameter Large Language Models (LLMs) for phishing detection. Through experiments on a curated dataset, we show that while LLMs currently underperform compared to ML and DL methods in terms of raw accuracy, they exhibit strong potential for identifying subtle, context-based phishing cues. We also investigate the impact of zero-shot and few-shot prompting strategies, revealing that LLM-rephrased emails can significantly degrade the performance of both ML and LLM-based detectors. Our benchmarking highlights that models like DeepSeek R1 Distill Qwen 14B (Q8_0) achieve competitive accuracy, above 80%, using only 17GB of VRAM, supporting their viability for cost-efficient deployment. We further assess the models' adversarial robustness and cost-performance tradeoffs, and demonstrate how lightweight LLMs can provide concise, interpretable explanations to support real-time decision-making. These findings position optimized LLMs as promising components in phishing defence systems and offer a path forward for integrating explainable, efficient AI into modern cybersecurity frameworks.

摘要: 网络钓鱼攻击变得越来越复杂，这凸显了对在高准确性和计算效率之间取得平衡的检测系统的需求。本文对传统机器学习（ML）、深度学习（DL）和用于网络钓鱼检测的量化小参数大型语言模型（LLM）进行了比较评估。通过对精心策划的数据集的实验，我们表明，虽然LLM目前在原始准确性方面表现不佳ML和DL方法，但它们在识别微妙的、基于上下文的网络钓鱼线索方面表现出强大的潜力。我们还研究了零激发和少激发策略的影响，揭示了LLM重新措辞的电子邮件会显着降低ML和基于LLM的检测器的性能。我们的基准测试强调，DeepSeek R1 Distill Qwen 14 B（Q8_0）等型号仅使用17 GB VRAM即可实现80%以上的竞争准确性，支持其具有成本效益的部署可行性。我们进一步评估了模型的对抗稳健性和成本-性能权衡，并展示了轻量级LLM如何提供简洁、可解释的解释以支持实时决策。这些发现将优化的LLM定位为网络钓鱼防御系统中有前途的组件，并为将可解释、高效的人工智能集成到现代网络安全框架中提供了前进的道路。



## **38. VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation**

Visual Trap：通过视觉基础操纵对图形用户界面代理进行秘密后门攻击 cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06899v1) [paper-pdf](http://arxiv.org/pdf/2507.06899v1)

**Authors**: Ziang Ye, Yang Zhang, Wentao Shi, Xiaoyu You, Fuli Feng, Tat-Seng Chua

**Abstract**: Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the visual grounding of GUI agent-mapping textual plans to GUI elements-can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent's behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose VisualTrap, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, e.g., being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents.

摘要: 由大型视觉语言模型（LVLM）驱动的图形用户界面（GUI）代理已经成为自动化人机交互的革命性方法，能够自主操作个人设备（例如，移动电话）或设备内的应用程序以类似于人类的方式执行复杂的现实世界任务。然而，它们与个人设备的紧密结合引发了重大的安全问题，包括后门攻击在内的许多威胁在很大程度上仍未得到解决。这项工作揭示了GUI代理的视觉基础-将文本计划映射到GUI元素-可以引入漏洞，从而实现新类型的后门攻击。通过针对视觉基础的后门攻击，即使给出了正确的任务解决计划，代理的行为也可能受到损害。为了验证此漏洞，我们提出了Visual Trap，这是一种可以通过误导代理定位文本计划来触发位置而不是预期目标来劫持接地的方法。Visual Trap使用注入有毒数据进行攻击的常见方法，并在视觉基础的预训练期间这样做，以确保攻击的实际可行性。经验结果表明，Visual Trap可以通过低至5%的有毒数据和高度隐蔽的视觉触发器（人眼看不见）有效劫持视觉基础;并且即使经过彻底的微调，攻击也可以推广到下游任务。此外，注入的触发器可以在不同的图形用户界面环境中保持有效，例如，正在接受移动/网络培训并推广到桌面环境。这些发现凸显了对图形用户界面代理后门攻击风险进行进一步研究的迫切性。



## **39. RAG Safety: Exploring Knowledge Poisoning Attacks to Retrieval-Augmented Generation**

RAG安全：探索对检索增强一代的知识中毒攻击 cs.CR

13 pages, 6 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.08862v1) [paper-pdf](http://arxiv.org/pdf/2507.08862v1)

**Authors**: Tianzhe Zhao, Jiaoyan Chen, Yanchi Ru, Haiping Zhu, Nan Hu, Jun Liu, Qika Lin

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving external data to mitigate hallucinations and outdated knowledge issues. Benefiting from the strong ability in facilitating diverse data sources and supporting faithful reasoning, knowledge graphs (KGs) have been increasingly adopted in RAG systems, giving rise to KG-based RAG (KG-RAG) methods. Though RAG systems are widely applied in various applications, recent studies have also revealed its vulnerabilities to data poisoning attacks, where malicious information injected into external knowledge sources can mislead the system into producing incorrect or harmful responses. However, these studies focus exclusively on RAG systems using unstructured textual data sources, leaving the security risks of KG-RAG largely unexplored, despite the fact that KGs present unique vulnerabilities due to their structured and editable nature. In this work, we conduct the first systematic investigation of the security issue of KG-RAG methods through data poisoning attacks. To this end, we introduce a practical, stealthy attack setting that aligns with real-world implementation. We propose an attack strategy that first identifies adversarial target answers and then inserts perturbation triples to complete misleading inference chains in the KG, increasing the likelihood that KG-RAG methods retrieve and rely on these perturbations during generation. Through extensive experiments on two benchmarks and four recent KG-RAG methods, our attack strategy demonstrates strong effectiveness in degrading KG-RAG performance, even with minimal KG perturbations. In-depth analyses are also conducted to understand the safety threats within the internal stages of KG-RAG systems and to explore the robustness of LLMs against adversarial knowledge.

摘要: 检索增强生成（RAG）通过检索外部数据来增强大型语言模型（LLM），以减轻幻觉和过时的知识问题。得益于知识图在促进多样化数据源和支持可信推理方面的强大能力，知识图在RAG系统中得到越来越多的应用，从而产生了基于知识图的RAG（KG-RAG）方法。虽然RAG系统被广泛应用于各种应用中，但最近的研究也揭示了其对数据中毒攻击的脆弱性，其中恶意信息注入外部知识源可以误导系统产生错误或有害的响应。然而，这些研究仅关注使用非结构化文本数据源的RAG系统，使得KG-RAG的安全风险在很大程度上未被探索，尽管KG由于其结构化和可编辑的性质而存在独特的漏洞。在这项工作中，我们通过数据中毒攻击对KG-RAG方法的安全问题进行了首次系统调查。为此，我们引入了一种与现实世界实现保持一致的实用、隐蔽的攻击设置。我们提出了一种攻击策略，首先识别对抗性目标答案，然后插入扰动三重组以完成KG中的误导性推理链，从而增加了KG-RAG方法在生成期间检索和依赖这些扰动的可能性。通过对两个基准测试和四种最近的KG-RAG方法的广泛实验，我们的攻击策略证明了即使在最小的KG扰动下，也可以有效降低KG-RAG性能。还进行了深入的分析，以了解KG-RAG系统内部阶段的安全威胁，并探索LLM针对对抗性知识的稳健性。



## **40. GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods**

GuidedBench：衡量和减轻野外LLM越狱方法的评估差异 cs.CL

Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2502.16903v2) [paper-pdf](http://arxiv.org/pdf/2502.16903v2)

**Authors**: Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Despite the growing interest in jailbreak methods as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. We conduct a systematic measurement study based on 37 jailbreak studies since 2022, focusing on both the methods and the evaluation systems they employ. We find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. This paper advocates a shift to a more nuanced, case-by-case evaluation paradigm. We introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset, detailed case-by-case evaluation guidelines and an evaluation system integrated with these guidelines -- GuidedEval. Experiments demonstrate that GuidedBench offers more accurate measurements of jailbreak performance, enabling meaningful comparisons across methods and uncovering new insights overlooked in previous evaluations. GuidedEval reduces inter-evaluator variance by at least 76.03\%. Furthermore, we observe that incorporating guidelines can enhance the effectiveness of jailbreak methods themselves, offering new insights into both attack strategies and evaluation paradigms.

摘要: 尽管人们越来越感兴趣越狱方法作为构建安全且负责任的大型语言模型（LLM）的有效红色团队工具，但有缺陷的评估系统设计导致其有效性评估存在显着差异。我们根据2022年以来的37项越狱研究进行了系统性的测量研究，重点关注他们采用的方法和评估体系。我们发现现有的评估系统缺乏针对具体案例的标准，导致对其有效性和安全性影响得出误导性结论。本文主张转向更加细致入微的逐个案例评估范式。我们引入了GuidedBench，这是一个新颖的基准，包括精心策划的有害问题数据集、详细的个案评估指南以及与这些指南集成的评估系统-- GuidedEval。实验表明，GuidedBench提供了更准确的越狱表现测量，能够进行各种方法之间有意义的比较，并发现之前评估中忽视的新见解。GuidedEval将评估者间方差减少至少76.03%。此外，我们观察到，纳入指南可以提高越狱方法本身的有效性，为攻击策略和评估范式提供新的见解。



## **41. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

尾部感知对抗攻击：高效LLM越狱的分布式方法 cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.04446v2) [paper-pdf](http://arxiv.org/pdf/2507.04446v2)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.

摘要: 为了保证大规模安全、稳健地部署大型语言模型（LLM），准确评估其对抗稳健性至关重要。现有的对抗性攻击通常针对单点贪婪世代的有害响应，忽视了LLM固有的随机性。在本文中，我们提出了一种新颖的对抗稳健性评估框架，该框架对整个输出分布（包括尾部风险）进行显式建模，为模型大规模稳健性提供更好的估计。通过将攻击过程描述为优化和采样之间的资源分配问题，我们确定了计算最优权衡，并表明将采样集成到现有攻击中可将ASB提高高达48%，并将效率提高高达两个数量级。我们的框架还使我们能够分析不同的攻击算法如何影响输出伤害分布。令人惊讶的是，我们发现大多数优化策略对输出危害影响很小。最后，我们引入了一个基于熵最大化的无数据概念验证目标，以演示我们的尾部感知视角如何实现新的优化目标。总体而言，我们的研究结果强调了尾部感知攻击和评估协议对于准确评估和加强LLM安全性的重要性。



## **42. An attention-aware GNN-based input defender against multi-turn jailbreak on LLMs**

一个具有注意力的基于GNN的输入防御者，防止LLM上的多回合越狱 cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.07146v1) [paper-pdf](http://arxiv.org/pdf/2507.07146v1)

**Authors**: Zixuan Huang, Kecheng Huang, Lihao Yin, Bowei He, Huiling Zhen, Mingxuan Yuan, Zili Shao

**Abstract**: Large Language Models (LLMs) have gained widespread popularity and are increasingly integrated into various applications. However, their capabilities can be exploited for both benign and harmful purposes. Despite rigorous training and fine-tuning for safety, LLMs remain vulnerable to jailbreak attacks. Recently, multi-turn attacks have emerged, exacerbating the issue. Unlike single-turn attacks, multi-turn attacks gradually escalate the dialogue, making them more difficult to detect and mitigate, even after they are identified.   In this study, we propose G-Guard, an innovative attention-aware GNN-based input classifier designed to defend against multi-turn jailbreak attacks on LLMs. G-Guard constructs an entity graph for multi-turn queries, explicitly capturing relationships between harmful keywords and queries even when those keywords appear only in previous queries. Additionally, we introduce an attention-aware augmentation mechanism that retrieves the most similar single-turn query based on the multi-turn conversation. This retrieved query is treated as a labeled node in the graph, enhancing the ability of GNN to classify whether the current query is harmful. Evaluation results demonstrate that G-Guard outperforms all baselines across all datasets and evaluation metrics.

摘要: 大型语言模型（LLM）已获得广泛流行，并越来越多地集成到各种应用程序中。然而，它们的能力可以被用于良性和有害的目的。尽管经过严格的培训和安全调整，LLM仍然容易受到越狱攻击。最近，出现了多回合攻击，加剧了这一问题。与单轮攻击不同，多轮攻击会逐渐升级对话，使其更难被发现和缓解，即使在被发现之后。   在这项研究中，我们提出了G-Guard，这是一种创新的基于注意力感知GNN的输入分类器，旨在抵御对LLM的多回合越狱攻击。G-Guard为多轮查询构建了一个实体图，显式地捕获有害关键字和查询之间的关系，即使这些关键字只出现在以前的查询中。此外，我们引入了一个注意力感知的增强机制，检索最相似的单轮查询的基础上的多轮对话。这个检索到的查询被视为图中的标签节点，增强了GNN对当前查询是否有害进行分类的能力。评估结果表明，G-Guard在所有数据集和评估指标中的表现优于所有基线。



## **43. Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions**

评估和改进大型语言模型的鲁棒性：调查和未来方向 cs.CL

33 pages, 5 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.11111v2) [paper-pdf](http://arxiv.org/pdf/2506.11111v2)

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.

摘要: 近年来，大型语言模型（LLM）因其理解和生成自然语言的能力而受到了广泛关注。随着快速发展和广泛应用（例如，代理人，联合情报），LLM的稳健性受到了越来越多的关注。作为许多人工智能应用的核心大脑，LLM的稳健性要求模型不仅要生成一致的内容，还要在处理意外的应用场景（例如，有毒提示、有限的噪音域数据、向外分布（OOD）应用程序等）。在这篇调查论文中，我们对LLM的稳健性进行了彻底的审查，旨在提供该领域的全面概念和方法术语并促进社区发展。具体来说，我们首先给出了LLM稳健性的正式定义，并给出了这篇调查论文的收集协议。然后，根据受干扰的输入类型，我们从以下角度组织本次调查：1）对抗稳健性：解决提示被故意操纵的问题，例如噪音提示、长上下文、数据攻击等; 2）OOD稳健性：处理意想不到的现实世界应用场景，例如OOD检测、零镜头传输、幻觉等; 3）稳健性评估：总结用于验证LLM稳健性的新评估数据集、指标和工具。在从各个角度回顾了代表性作品后，我们讨论并强调了该领域未来的机会和研究方向。同时，我们还组织相关工作并提供易于搜索的项目（https：//github.com/zhangkunzk/Awesome-LLM-Robustness-papers）来支持社区。



## **44. Breaking PEFT Limitations: Leveraging Weak-to-Strong Knowledge Transfer for Backdoor Attacks in LLMs**

打破PEFT限制：利用弱到强的知识转移进行LLM中的后门攻击 cs.CR

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2409.17946v4) [paper-pdf](http://arxiv.org/pdf/2409.17946v4)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Yanhao Jia, Luwei Xiao, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning (FPFT). However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from the weak-to-strong based on Feature Alignment-enhanced Knowledge Distillation (FAKD). Specifically, we poison small-scale language models through FPFT to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through FAKD, which employs PEFT. Theoretical analysis reveals that FAKD has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of FAKD on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.

摘要: 尽管大型语言模型（LLM）因其卓越的功能而被广泛应用，但已被证明容易受到后门攻击。这些攻击通过毒害训练样本和全参数微调（FPFT）将有针对性的漏洞引入LLM。然而，这种后门攻击是有限的，因为它们需要大量的计算资源，特别是随着LLM规模的增加。此外，参数高效微调（PEFT）提供了一种替代方案，但受限制的参数更新可能会阻碍触发器与目标标签的对齐。在这项研究中，我们首先验证了使用PEFT进行的后门攻击在实现可行的性能时可能会遇到挑战。为了解决这些问题并提高PEFT后门攻击的有效性，我们提出了一种基于特征对齐增强知识提炼（FAKD）的从弱到强的新型后门攻击算法。具体来说，我们通过FPFT毒害小规模语言模型，以充当教师模型。然后，教师模式通过采用PEFT的FAKD秘密地将后门转移到大规模学生模式。理论分析表明，FAKD有潜力增强后门攻击的有效性。我们展示了FAKD在四种语言模型、四种后门攻击算法和两种不同的教师模型架构上的分类任务上的卓越性能。实验结果表明，针对PEFT的后门攻击成功率接近100%。



## **45. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

21 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2411.08003v2) [paper-pdf](http://arxiv.org/pdf/2411.08003v2)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.

摘要: 在对抗环境中（例如网络攻击和虚假信息攻击）对大型语言模型（LLM）的输出进行归因会带来重大挑战，而且其重要性可能会越来越大。我们从理论和实证的角度来处理这个归因问题，借鉴形式语言理论（极限识别）和对不断扩大的LLM生态系统的数据驱动分析。通过将LLM的一组可能输出建模为形式语言，我们分析有限的文本样本是否可以唯一地确定原始模型。我们的结果表明，在模型之间能力重叠的温和假设下，某些类别的LLM从根本上无法仅从其输出中识别。我们描绘了理论可识别性的四种制度：（1）无限一类确定性（离散）LLM语言不可识别（Gold的经典结果来自1967年）;（2）无限类概率LLM也是不可识别的（通过确定性情况的扩展）;（3）有限类确定性LLM是可识别的（与Angluin的泄密标准一致）;以及（4）即使是有限类的概率LLM也可能是不可识别的（我们提供了一个新的反例来建立这个负结果）。作为对这些理论见解的补充，我们量化了近年来给定输出的合理模型起源（假设空间）数量的爆炸式增长。即使在保守的假设下--每个开源模型最多在一个新厕所上进行微调--不同候选模型的数量也大约每0.5年翻一番，并且允许多数据集微调组合可以产生翻倍的时间短至0.28年。这种组合增长，加上所有模型和潜在用户的暴力可能性归因的非凡计算成本，使得详尽的归因在实践中不可行。



## **46. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06489v1) [paper-pdf](http://arxiv.org/pdf/2507.06489v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于部署LLM至关重要，以确保许多高风险应用程序中人机交互的透明度、信任和安全。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们引入了一个新颖的框架，通过干扰和基于越狱的方法攻击言语信心分数，并表明这些攻击可能会显着危及言语信心估计并导致答案频繁变化。我们研究了各种提示策略、模型大小和应用领域，揭示了当前的信心激发方法很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了迫切需要设计更强大的机制来表达LLM的信心，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **47. Bridging AI and Software Security: A Comparative Vulnerability Assessment of LLM Agent Deployment Paradigms**

连接人工智能和软件安全：LLM代理部署范式的比较漏洞评估 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06323v1) [paper-pdf](http://arxiv.org/pdf/2507.06323v1)

**Authors**: Tarek Gasmi, Ramzi Guesmi, Ines Belhadj, Jihene Bennaceur

**Abstract**: Large Language Model (LLM) agents face security vulnerabilities spanning AI-specific and traditional software domains, yet current research addresses these separately. This study bridges this gap through comparative evaluation of Function Calling architecture and Model Context Protocol (MCP) deployment paradigms using a unified threat classification framework. We tested 3,250 attack scenarios across seven language models, evaluating simple, composed, and chained attacks targeting both AI-specific threats (prompt injection) and software vulnerabilities (JSON injection, denial-of-service). Function Calling showed higher overall attack success rates (73.5% vs 62.59% for MCP), with greater system-centric vulnerability while MCP exhibited increased LLM-centric exposure. Attack complexity dramatically amplified effectiveness, with chained attacks achieving 91-96% success rates. Counterintuitively, advanced reasoning models demonstrated higher exploitability despite better threat detection. Results demonstrate that architectural choices fundamentally reshape threat landscapes. This work establishes methodological foundations for cross-domain LLM agent security assessment and provides evidence-based guidance for secure deployment. Code and experimental materials are available at https: // github. com/ theconsciouslab-ai/llm-agent-security.

摘要: 大型语言模型（LLM）代理面临跨越人工智能特定和传统软件领域的安全漏洞，但当前的研究分别解决了这些问题。本研究通过使用统一的威胁分类框架对功能调用架构和模型上下文协议（HCP）部署范式进行比较评估来弥合这一差距。我们测试了七种语言模型中的3，250个攻击场景，评估了针对人工智能特定威胁（提示注入）和软件漏洞（SON注入、拒绝服务）的简单、组合和连锁攻击。功能调用显示出更高的总体攻击成功率（73.5% vs 62.59%），以系统为中心的脆弱性更大，而麦克唐纳则显示出以LLM为中心的暴露率增加。攻击的复杂性极大地提高了有效性，连锁攻击的成功率达到了91-96%。与直觉相反，尽管威胁检测更好，但高级推理模型仍表现出更高的可利用性。结果表明，架构选择从根本上重塑了威胁格局。这项工作为跨域LLM代理安全评估奠定了方法论基础，并为安全部署提供了基于证据的指导。代码和实验材料可在https：// github上获取。com/ theconsciouslab-ai/llm-Agent-secure。



## **48. CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations**

CAVGAN：通过对其内部代表的生成性对抗攻击统一LLM的越狱和辩护 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06043v1) [paper-pdf](http://arxiv.org/pdf/2507.06043v1)

**Authors**: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

**Abstract**: Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.

摘要: 安全对齐使大型语言模型（LLM）能够获得针对恶意查询的保护，但各种越狱攻击方法揭示了这种安全机制的漏洞。之前的研究已经孤立了LLM越狱攻击和防御。我们分析了LLM的安全保护机制，提出了攻击与防御相结合的框架。我们的方法基于LLM中间层嵌入的线性可分离性质，以及越狱攻击的本质，旨在嵌入有害问题并将其转移到安全区域。我们利用生成对抗网络（GAN）来学习LLM内部的安全判断边界，以实现高效的越狱攻击和防御。实验结果表明，我们的方法在三种流行的LLM中平均越狱成功率为88.85%，而在最先进的越狱数据集上的防御成功率平均达到84.17%。这不仅验证了我们方法的有效性，还揭示了LLM的内部安全机制，为增强模型安全性提供了新的见解。代码和数据可在https://github.com/NLPGM/CAVGAN上获取。



## **49. Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks**

增强LLM水印针对擦除和欺骗攻击的弹性 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06274v1) [paper-pdf](http://arxiv.org/pdf/2507.06274v1)

**Authors**: Huanming Shen, Baizhou Huang, Xiaojun Wan

**Abstract**: Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.

摘要: 水印是防止大型语言模型（LLM）滥用的一种有希望的防御方法，但它仍然容易受到擦洗和欺骗攻击。该漏洞源于由水印窗口大小决定的固有权衡：较小的窗口更难抵抗擦洗，但更容易进行反向工程，从而实现低成本的基于统计学的欺骗攻击。这项工作通过引入一种新颖的机制（等效纹理密钥）打破了这种权衡，其中水印窗口内的多个令牌可以独立支持检测。基于冗余度，我们提出了一种新的子词汇分解等效tExture密钥（SEEK）水印方案。它实现了帕累托改进，提高了针对擦除攻击的弹性，而不会损害欺骗的稳健性。实验证明了SEEK相对于现有方法的优越性，在不同的数据集设置中产生了+88.2%/+92.3%/+82.0%的欺骗鲁棒性收益，并产生了+10.2%/+6.4%/+24.6%的擦洗鲁棒性收益。



## **50. ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis**

ETrace：通过基于LLM的跟踪分析在智能合同中进行事件驱动的漏洞检测 cs.CR

4 pages, 1 figure. Submitted to the 16th Asia-Pacific Symposium on  Internetware (Internetware 2025)

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2506.15790v2) [paper-pdf](http://arxiv.org/pdf/2506.15790v2)

**Authors**: Chenyang Peng, Haijun Wang, Yin Wu, Hao Wu, Ming Fan, Yitao Zhao, Ting Liu

**Abstract**: With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.

摘要: 随着区块链技术在各个领域的深入应用，确保智能合约的安全性和稳定性已成为一项严峻的挑战。当前漏洞检测中的安全分析方法可以分为静态分析和动态分析方法。然而，这些现有的传统漏洞检测方法主要依赖于分析原始合同代码，并非所有智能合同都提供可访问代码。我们提出ETrace，一种新颖的事件驱动的智能合同漏洞检测框架，它通过LLM支持的跟踪分析来唯一地识别潜在漏洞，而无需访问源代码。通过从事务日志中提取细粒度事件序列，该框架利用大型语言模型（LLM）作为自适应语义解释器，通过思想链推理重建事件分析。ETrace实现模式匹配，以建立事务行为模式和已知攻击行为之间的因果联系。此外，我们通过初步实验结果验证了ETrace的有效性。



