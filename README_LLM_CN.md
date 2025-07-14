# Latest Large Language Model Attack Papers
**update at 2025-07-14 09:57:09**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2401.17256v4) [paper-pdf](http://arxiv.org/pdf/2401.17256v4)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **2. A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1**

基于LLM的论点分类的全面研究：从LLAMA到GPT-4 o到Deepseek-R1 cs.CL

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08621v1) [paper-pdf](http://arxiv.org/pdf/2507.08621v1)

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.

摘要: 论据挖掘（AM）是一个跨学科研究领域，集成了逻辑、哲学、语言学、修辞学、法学、心理学和计算机科学的见解。它涉及自动识别和提取论点成分（例如前提和主张），以及检测它们之间的关系（例如支持、攻击或中立）。最近，该领域取得了显着的进步，特别是随着大型语言模型（LLM）的出现，与传统方法和其他深度学习模型相比，LLM提高了分析和提取参数语义的效率。有许多用于测试和验证LLM质量的基准，但在公开可用的论点分类数据库中仍然缺乏有关这些模型操作的研究和结果。本文使用Args.me和UKP等不同数据集对LLM进行了一项研究。测试的模型包括GPT、Llama和DeepSeek的版本，以及包含思想链算法的推理增强变体。结果表明，ChatGPT-4 o在论点分类基准方面优于其他。在包含推理能力的模型中，Deepseek-R1显示出其优越性。然而，尽管具有优势，GPT-4 o和Deepseek-R1仍然会犯错误。讨论了所有模型最常见的错误。据我们所知，所介绍的工作是首次使用LLM和提示算法对上述数据集进行更广泛的分析。该工作还揭示了已知提示算法在论据分析中的一些弱点，同时指出了改进的方向。这项工作的附加值是对可用论点数据集的深入分析并展示其缺点。



## **3. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

LLM基于代理的完全计算机接管攻击的阴暗面 cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.06850v3) [paper-pdf](http://arxiv.org/pdf/2507.06850v3)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.

摘要: 大型语言模型（LLM）代理和多代理系统的快速采用使自然语言处理和生成具有前所未有的能力。然而，这些系统引入了前所未有的安全漏洞，超出了传统的即时注入攻击的范围。本文首次对LLM代理进行了全面评估，作为攻击载体，这些攻击载体能够通过利用自主实体相互交互和影响的代理人工智能系统内的信任边界来实现完全的计算机接管。我们证明，对手可以利用三种不同的攻击表面--直接提示注入、RAG后门攻击和代理间信任利用--来强迫流行的LLM（包括GPT-4 o、Claude-4和Gemini-2.5）在受害者机器上自主安装和执行恶意软件。我们对17个最先进的LLM的评估揭示了一个令人震惊的漏洞层次结构：虽然41.2%的模型屈服于直接即时注入，但52.9%的模型容易受到RAG后门攻击，并且关键的82.4%可以通过代理间信任利用而受到损害。值得注意的是，我们发现成功抵抗直接恶意命令的LLM将在对等代理请求时执行相同的有效负载，这揭示了当前多代理安全模型中的一个根本缺陷。我们的研究结果表明，只有5.9%的测试模型（1/17）被证明能够抵抗所有攻击载体，其中大多数表现出依赖于上下文的安全行为，从而创建了可利用的盲点。我们的研究结果还强调了提高对LLM安全风险的认识和研究的必要性，这表明网络安全威胁的范式转变，人工智能工具本身成为复杂的攻击载体。



## **4. Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection**

Emoji攻击：增强针对LLM法官检测的越狱攻击 cs.CL

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2411.01077v4) [paper-pdf](http://arxiv.org/pdf/2411.01077v4)

**Authors**: Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson

**Abstract**: Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.

摘要: 越狱技术欺骗大型语言模型（LLM）产生受限输出，构成潜在威胁。一种防御措施是使用另一位LLM作为法官来评估生成文本的危害性。然而，我们发现这些Judge LLM很容易受到标记分割偏见的影响，当分隔符改变标记化过程、将单词分割成更小的子标记时，就会出现这个问题。这会改变整个序列的嵌入，降低检测准确性，并允许有害内容被错误分类为安全内容。在本文中，我们介绍了Emoji Attack，这是一种新颖的策略，通过利用代币分割偏见来放大现有的越狱提示。我们的方法利用上下文学习，在LLM法官评估文本之前系统地将表情符号插入文本中，从而引发嵌入失真，从而显着降低检测到不安全内容的可能性。与传统的分隔符不同，表情符号还会引入语义歧义，使它们在这种攻击中特别有效。通过对最先进的Judge LLM的实验，我们证明Emoji Attack大幅降低了不安全的预测率，绕过了现有的保障措施。



## **5. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

从心理测量学角度通过攻击来评估大型语言模型中的内隐偏差 cs.CL

Accepted to ACL 2025 Findings

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2406.14023v5) [paper-pdf](http://arxiv.org/pdf/2406.14023v5)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.

摘要: 随着大型语言模型（LLM）成为信息获取的重要方式，人们越来越担心LLM可能会加剧不道德内容的传播，包括在没有明确有害词语的情况下伤害某些人群的隐性偏见。在本文中，我们进行了严格的评估LLM的隐性偏见对某些人口统计数据的攻击，从心理测量学的角度，以引起有偏见的观点的协议。受认知和社会心理学中心理测量原则的启发，我们提出了三种攻击方法，即伪装、欺骗和教导。综合相应的攻击指令，我们构建了两个基准：（1）双语数据集，其中包含涵盖四种偏见类型（2.7 K实例）的偏见陈述，用于广泛的比较分析，和（2）BUMBLE，一个跨越九种常见偏见类型（12.7 K实例）的更大基准，用于全面评估。对流行的商业和开源LLM的广泛评估表明，我们的方法比竞争基线更有效地引发LLM的内部偏见。我们的攻击方法和基准提供了评估LLM道德风险的有效手段，推动LLM在开发过程中实现更强的问责制。我们的代码、数据和基准可在https://yuchenwen1.github.io/ImplicitBiasEvaluation/上获取。



## **6. Invariant-based Robust Weights Watermark for Large Language Models**

大型语言模型的基于不变的鲁棒权重水印 cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08288v1) [paper-pdf](http://arxiv.org/pdf/2507.08288v1)

**Authors**: Qingxiao Guo, Xinjie Zhu, Yilong Ma, Hui Jin, Yunhao Wang, Weifeng Zhang, Xiaobing Guo

**Abstract**: Watermarking technology has gained significant attention due to the increasing importance of intellectual property (IP) rights, particularly with the growing deployment of large language models (LLMs) on billions resource-constrained edge devices. To counter the potential threats of IP theft by malicious users, this paper introduces a robust watermarking scheme without retraining or fine-tuning for transformer models. The scheme generates a unique key for each user and derives a stable watermark value by solving linear constraints constructed from model invariants. Moreover, this technology utilizes noise mechanism to hide watermark locations in multi-user scenarios against collusion attack. This paper evaluates the approach on three popular models (Llama3, Phi3, Gemma), and the experimental results confirm the strong robustness across a range of attack methods (fine-tuning, pruning, quantization, permutation, scaling, reversible matrix and collusion attacks).

摘要: 由于知识产权（IP）权的重要性日益增加，特别是随着大型语言模型（LLM）在数十亿个资源有限的边缘设备上部署的日益增多，水印技术受到了广泛关注。为了应对恶意用户IP盗窃的潜在威胁，本文引入了一种鲁棒的水印方案，无需对Transformer模型进行再培训或微调。该方案为每个用户生成唯一的密钥，并通过求解由模型不变量构建的线性约束来推导稳定的水印值。此外，该技术利用噪音机制来隐藏多用户场景中的水印位置，以防止共谋攻击。本文在三种流行模型（Llama 3、Phi 3、Gemma）上评估了该方法，实验结果证实了一系列攻击方法（微调、修剪、量化、置换、缩放、可逆矩阵和共谋攻击）的强大鲁棒性。



## **7. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

突破安全极限：2025年ATLAS挑战赛技术报告 cs.CR

AdvML@CVPR Challenge Report

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2506.12430v2) [paper-pdf](http://arxiv.org/pdf/2506.12430v2)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.

摘要: 多模式大型语言模型（MLLM）在不同的应用程序中实现了变革性的进步，但仍然容易受到安全威胁，尤其是引发有害输出的越狱攻击。为了系统地评估和提高其安全性，我们组织了对抗性测试和大模型对齐安全大挑战赛（ATLAS）2025。本技术报告介绍了比赛的结果，其中86个团队通过对抗性图像文本攻击分两个阶段测试MLLM漏洞：白盒和黑匣子评估。竞赛结果凸显了确保MLLM方面持续存在的挑战，并为开发更强大的防御机制提供了宝贵的指导。该挑战为MLLM安全评估建立了新的基准，并为推进更安全的多模式人工智能系统奠定了基础。此挑战的代码和数据可在https://github.com/NY1024/ATLAS_Challenge_2025上公开获取。



## **8. A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking**

动态Stackelberg游戏框架，用于针对LLM越狱的大型人工智能防御 cs.AI

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08207v1) [paper-pdf](http://arxiv.org/pdf/2507.08207v1)

**Authors**: Zhengye Han, Quanyan Zhu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking.

摘要: 随着大型语言模型（LLM）越来越多地部署在关键应用程序中，越狱的挑战（对手操纵模型以绕过安全机制）已成为一个重大问题。本文提出了一个动态Stackelberg博弈框架，来建模LLM越狱背景下攻击者和防御者之间的互动。该框架将预算-响应动态视为一个顺序扩展形式的游戏，其中防御者作为领导者，承诺采取策略，同时预测攻击者的最佳响应。我们提出了一种新型的代理人工智能解决方案，即“紫色代理”，它使用快速探索随机树（RTI）集成了对抗性探索和防御策略。Purple Agent主动模拟潜在的攻击轨迹，并主动干预以防止有害输出。这种方法提供了一种分析对抗动态的原则性方法，并为减轻越狱风险提供了基础。



## **9. Beyond the Worst Case: Extending Differential Privacy Guarantees to Realistic Adversaries**

超越最坏情况：将差异隐私保证扩展到现实对手 cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08158v1) [paper-pdf](http://arxiv.org/pdf/2507.08158v1)

**Authors**: Marika Swanberg, Meenatchi Sundaram Muthu Selva Annamalai, Jamie Hayes, Borja Balle, Adam Smith

**Abstract**: Differential Privacy (DP) is a family of definitions that bound the worst-case privacy leakage of a mechanism. One important feature of the worst-case DP guarantee is it naturally implies protections against adversaries with less prior information, more sophisticated attack goals, and complex measures of a successful attack. However, the analytical tradeoffs between the adversarial model and the privacy protections conferred by DP are not well understood thus far. To that end, this work sheds light on what the worst-case guarantee of DP implies about the success of attackers that are more representative of real-world privacy risks.   In this paper, we present a single flexible framework that generalizes and extends the patchwork of bounds on DP mechanisms found in prior work. Our framework allows us to compute high-probability guarantees for DP mechanisms on a large family of natural attack settings that previous bounds do not capture. One class of such settings is the approximate reconstruction of multiple individuals' data, such as inferring nearly entire columns of a tabular data set from noisy marginals and extracting sensitive information from DP-trained language models.   We conduct two empirical case studies to illustrate the versatility of our bounds and compare them to the success of state-of-the-art attacks. Specifically, we study attacks that extract non-uniform PII from a DP-trained language model, as well as multi-column reconstruction attacks where the adversary has access to some columns in the clear and attempts to reconstruct the remaining columns for each person's record. We find that the absolute privacy risk of attacking non-uniform data is highly dependent on the adversary's prior probability of success. Our high probability bounds give us a nuanced understanding of the privacy leakage of DP mechanisms in a variety of previously understudied attack settings.

摘要: 差异隐私（DP）是一系列定义，限制了机制的最坏情况隐私泄露。最坏情况DP保证的一个重要特征是，它自然意味着针对先验信息较少、攻击目标更复杂且成功攻击措施复杂的对手提供保护。然而，迄今为止，对抗模型和DP赋予的隐私保护之间的分析权衡还没有得到很好的理解。为此，这项工作揭示了DP的最坏情况保证对更能代表现实世界隐私风险的攻击者的成功意味着什么。   在本文中，我们提出了一个灵活的框架，该框架概括和扩展了先前工作中发现的DP机制边界的拼凑。我们的框架允许我们在以前的界限无法捕捉的一大系列自然攻击设置上计算DP机制的高概率保证。一类此类设置是多个人数据的大致重建，例如从有噪的边缘推断表格数据集的几乎整个列，并从DP训练的语言模型中提取敏感信息。   我们进行了两个实证案例研究，以说明我们边界的多功能性，并将它们与最先进的攻击的成功进行比较。具体来说，我们研究了从DP训练的语言模型中提取非均匀PRI的攻击，以及多列重建攻击，其中对手可以以明文方式访问某些列并试图为每个人的记录重建剩余列。我们发现，攻击非均匀数据的绝对隐私风险高度取决于对手的先验成功概率。我们的高概率界限让我们对各种以前未充分研究的攻击环境中DP机制的隐私泄露有了细致入微的了解。



## **10. Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)**

为Red-Teaming大型语言模型（LLM）操作威胁模型 cs.CL

Transactions of Machine Learning Research (TMLR)

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2407.14937v2) [paper-pdf](http://arxiv.org/pdf/2407.14937v2)

**Authors**: Apurv Verma, Satyapriya Krishna, Sebastian Gehrmann, Madhavan Seshadri, Anu Pradhan, Tom Ault, Leslie Barrett, David Rabinowitz, John Doucette, NhatHai Phan

**Abstract**: Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.

摘要: 使用大型语言模型（LLM）创建安全且有弹性的应用程序需要预测、调整和应对不可预见的威胁。红色团队已成为识别现实世界LLM实施中漏洞的关键技术。本文提出了一个详细的威胁模型，并提供了对LLM的红色团队攻击的知识系统化（SoK）。我们根据LLM开发和部署过程的阶段开发攻击分类，并从之前的研究中提取各种见解。此外，我们还为从业者编写了防御方法和实用的红色团队策略。通过描述突出的攻击主题并揭示各种切入点，本文提供了一个框架来提高基于LLM的系统的安全性和稳健性。



## **11. Defending Against Prompt Injection With a Few DefensiveTokens**

使用一些防御代币来防御即时注射 cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07974v1) [paper-pdf](http://arxiv.org/pdf/2507.07974v1)

**Authors**: Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, David Wagner

**Abstract**: When large language model (LLM) systems interact with external data to perform complex tasks, a new attack, namely prompt injection, becomes a significant threat. By injecting instructions into the data accessed by the system, the attacker is able to override the initial user task with an arbitrary task directed by the attacker. To secure the system, test-time defenses, e.g., defensive prompting, have been proposed for system developers to attain security only when needed in a flexible manner. However, they are much less effective than training-time defenses that change the model parameters. Motivated by this, we propose DefensiveToken, a test-time defense with prompt injection robustness comparable to training-time alternatives. DefensiveTokens are newly inserted as special tokens, whose embeddings are optimized for security. In security-sensitive cases, system developers can append a few DefensiveTokens before the LLM input to achieve security with a minimal utility drop. In scenarios where security is less of a concern, developers can simply skip DefensiveTokens; the LLM system remains the same as there is no defense, generating high-quality responses. Thus, DefensiveTokens, if released alongside the model, allow a flexible switch between the state-of-the-art (SOTA) utility and almost-SOTA security at test time. The code is available at https://github.com/Sizhe-Chen/DefensiveToken.

摘要: 当大型语言模型（LLM）系统与外部数据交互以执行复杂任务时，一种新的攻击（即提示注入）将成为重大威胁。通过将指令注入系统访问的数据中，攻击者能够用攻击者指示的任意任务覆盖初始用户任务。为了保护系统，测试时防御措施，例如防御性提示已被建议供系统开发人员仅在需要时以灵活的方式获得安全性。然而，它们比改变模型参数的训练时防御有效得多。出于此动机，我们提出了DefensiveToken，这是一种测试时防御，具有与训练时替代方案相当的即时注入鲁棒性。DefensiveTokens作为特殊令牌新插入，其嵌入针对安全性进行了优化。在安全敏感的情况下，系统开发人员可以在LLM输入之前添加一些DefensiveTokens，以最小的实用程序下降来实现安全性。在安全性不太值得关注的场景中，开发人员可以简单地跳过DefensiveTokens; LLM系统由于没有防御而保持不变，从而生成高质量的响应。因此，DefensiveTokens如果与该模型一起发布，将允许在测试时在最先进的（SOTA）实用程序和几乎SOTA安全性之间灵活切换。该代码可在https://github.com/Sizhe-Chen/DefensiveToken上获取。



## **12. Evaluating Robustness of Large Audio Language Models to Audio Injection: An Empirical Study**

评估大型音频语言模型对音频注入的稳健性：一项实证研究 cs.CL

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2505.19598v2) [paper-pdf](http://arxiv.org/pdf/2505.19598v2)

**Authors**: Guanyu Hou, Jiaming He, Yinhang Zhou, Ji Guo, Yitong Qiao, Rui Zhang, Wenbo Jiang

**Abstract**: Large Audio-Language Models (LALMs) are increasingly deployed in real-world applications, yet their robustness against malicious audio injection attacks remains underexplored. This study systematically evaluates five leading LALMs across four attack scenarios: Audio Interference Attack, Instruction Following Attack, Context Injection Attack, and Judgment Hijacking Attack. Using metrics like Defense Success Rate, Context Robustness Score, and Judgment Robustness Index, their vulnerabilities and resilience were quantitatively assessed. Experimental results reveal significant performance disparities among models; no single model consistently outperforms others across all attack types. The position of malicious content critically influences attack effectiveness, particularly when placed at the beginning of sequences. A negative correlation between instruction-following capability and robustness suggests models adhering strictly to instructions may be more susceptible, contrasting with greater resistance by safety-aligned models. Additionally, system prompts show mixed effectiveness, indicating the need for tailored strategies. This work introduces a benchmark framework and highlights the importance of integrating robustness into training pipelines. Findings emphasize developing multi-modal defenses and architectural designs that decouple capability from susceptibility for secure LALMs deployment.

摘要: 大型音频语言模型（LALM）越来越多地部署在现实世界的应用程序中，但它们针对恶意音频注入攻击的稳健性仍然没有得到充分的研究。本研究系统地评估了针对四种攻击场景的五种主要LALM：音频干扰攻击、指令跟随攻击、上下文注入攻击和判断劫持攻击。使用防御成功率、上下文稳健性得分和判断稳健性指数等指标，量化评估了他们的脆弱性和弹性。实验结果揭示了模型之间的显着性能差异;没有一个模型在所有攻击类型中始终优于其他模型。恶意内容的位置严重影响攻击效果，特别是当放置在序列的开头时。指令遵循能力和稳健性之间的负相关性表明，严格遵守指令的模型可能更容易受到影响，而安全一致的模型则具有更大的抵抗力。此外，系统提示显示出好坏参半的有效性，表明需要定制策略。这项工作引入了基准框架，并强调了将稳健性集成到训练管道中的重要性。研究结果强调开发多模式防御和架构设计，将能力与安全LALM部署的敏感性脱钩。



## **13. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

“我很坏”：在音频语言模型中解释秘密、普遍和稳健的音频越狱 cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2502.00718v2) [paper-pdf](http://arxiv.org/pdf/2502.00718v2)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.

摘要: 多模式大型语言模型的兴起引入了创新的人机交互范式，但也给机器学习安全带来了重大挑战。由于口语交流的直观性，音频语言模型（ILM）尤其重要，但人们对其失败模式知之甚少。本文探讨了针对ILM的音频越狱，重点关注它们绕过对齐机制的能力。我们构建了跨越提示、任务甚至基本音频样本的对抗性扰动，展示了音频模式中的第一次普遍越狱，并表明这些在模拟的现实世界条件下仍然有效。除了证明攻击可行性之外，我们还分析了ILM如何解释这些音频对抗示例，并揭示它们来编码难以察觉的第一人称有毒语音-这表明用于引发有毒输出的最有效的干扰专门嵌入了音频信号中的语言特征。这些结果对于理解多模式模型中不同模式之间的相互作用具有重要意义，并为增强对抗性音频攻击的防御提供了可行的见解。



## **14. GuardVal: Dynamic Large Language Model Jailbreak Evaluation for Comprehensive Safety Testing**

GuardVal：用于全面安全测试的动态大语言模型越狱评估 cs.LG

24 pages

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07735v1) [paper-pdf](http://arxiv.org/pdf/2507.07735v1)

**Authors**: Peiyan Zhang, Haibo Jin, Liying Kang, Haohan Wang

**Abstract**: Jailbreak attacks reveal critical vulnerabilities in Large Language Models (LLMs) by causing them to generate harmful or unethical content. Evaluating these threats is particularly challenging due to the evolving nature of LLMs and the sophistication required in effectively probing their vulnerabilities. Current benchmarks and evaluation methods struggle to fully address these challenges, leaving gaps in the assessment of LLM vulnerabilities. In this paper, we review existing jailbreak evaluation practices and identify three assumed desiderata for an effective jailbreak evaluation protocol. To address these challenges, we introduce GuardVal, a new evaluation protocol that dynamically generates and refines jailbreak prompts based on the defender LLM's state, providing a more accurate assessment of defender LLMs' capacity to handle safety-critical situations. Moreover, we propose a new optimization method that prevents stagnation during prompt refinement, ensuring the generation of increasingly effective jailbreak prompts that expose deeper weaknesses in the defender LLMs. We apply this protocol to a diverse set of models, from Mistral-7b to GPT-4, across 10 safety domains. Our findings highlight distinct behavioral patterns among the models, offering a comprehensive view of their robustness. Furthermore, our evaluation process deepens the understanding of LLM behavior, leading to insights that can inform future research and drive the development of more secure models.

摘要: 越狱攻击揭示了大型语言模型（LLM）中的关键漏洞，导致它们生成有害或不道德的内容。评估这些威胁是特别具有挑战性的，由于不断变化的性质LLM和复杂性需要有效地探测其漏洞。目前的基准和评估方法难以充分应对这些挑战，在评估LLM脆弱性方面留下了空白。在本文中，我们回顾了现有的越狱评估实践，并确定了三个假设的必要条件，一个有效的越狱评估协议。为了应对这些挑战，我们引入了GuardVal，这是一种新的评估协议，可以根据防守方LLM的状态动态生成和改进越狱提示，从而对防守方LLM处理安全关键情况的能力提供更准确的评估。此外，我们提出了一种新的优化方法，可以防止在即时改进期间出现停滞，确保生成越来越有效的越狱提示，从而暴露防御者LLM中更深层次的弱点。我们将该协议应用于10个安全领域的一系列不同模型，从Mistral-7 b到GPT-4。我们的研究结果强调了模型之间不同的行为模式，并全面了解其稳健性。此外，我们的评估过程加深了对LLM行为的理解，从而获得了可以为未来研究提供信息并推动更安全模型的开发的见解。



## **15. May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks**

请注意吗？使用架构感知攻击突破基于微调的提示注入防御 cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07417v1) [paper-pdf](http://arxiv.org/pdf/2507.07417v1)

**Authors**: Nishit V. Pandya, Andrey Labunets, Sicun Gao, Earlence Fernandes

**Abstract**: A popular class of defenses against prompt injection attacks on large language models (LLMs) relies on fine-tuning the model to separate instructions and data, so that the LLM does not follow instructions that might be present with data. There are several academic systems and production-level implementations of this idea. We evaluate the robustness of this class of prompt injection defenses in the whitebox setting by constructing strong optimization-based attacks and showing that the defenses do not provide the claimed security properties. Specifically, we construct a novel attention-based attack algorithm for text-based LLMs and apply it to two recent whitebox defenses SecAlign (CCS 2025) and StruQ (USENIX Security 2025), showing attacks with success rates of up to 70% with modest increase in attacker budget in terms of tokens. Our findings make fundamental progress towards understanding the robustness of prompt injection defenses in the whitebox setting. We release our code and attacks at https://github.com/nishitvp/better_opts_attacks

摘要: 针对大型语言模型（LLM）的即时注入攻击的一类流行防御依赖于对模型进行微调以分离指令和数据，以便LLM不会遵循可能存在于数据中的指令。这个想法有几个学术系统和生产级实现。我们通过构建强大的基于优化的攻击并表明这些防御不提供声称的安全属性来评估白盒设置中此类即时注入防御的稳健性。具体来说，我们为基于文本的LLM构建了一种新颖的基于注意力的攻击算法，并将其应用于最近的两种白盒防御SecAlign（CCCS 2025）和StruQ（USENIX Security 2025），显示攻击成功率高达70%，攻击者预算在代币方面略有增加。我们的研究结果在理解白盒环境中即时注射防御的稳健性方面取得了根本性进展。我们在https://github.com/nishitvp/better_opts_attacks上发布我们的代码和攻击



## **16. Hybrid LLM-Enhanced Intrusion Detection for Zero-Day Threats in IoT Networks**

针对物联网网络中零日威胁的混合LLM增强型入侵检测 cs.CR

6 pages, IEEE conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07413v1) [paper-pdf](http://arxiv.org/pdf/2507.07413v1)

**Authors**: Mohammad F. Al-Hammouri, Yazan Otoum, Rasha Atwa, Amiya Nayak

**Abstract**: This paper presents a novel approach to intrusion detection by integrating traditional signature-based methods with the contextual understanding capabilities of the GPT-2 Large Language Model (LLM). As cyber threats become increasingly sophisticated, particularly in distributed, heterogeneous, and resource-constrained environments such as those enabled by the Internet of Things (IoT), the need for dynamic and adaptive Intrusion Detection Systems (IDSs) becomes increasingly urgent. While traditional methods remain effective for detecting known threats, they often fail to recognize new and evolving attack patterns. In contrast, GPT-2 excels at processing unstructured data and identifying complex semantic relationships, making it well-suited to uncovering subtle, zero-day attack vectors. We propose a hybrid IDS framework that merges the robustness of signature-based techniques with the adaptability of GPT-2-driven semantic analysis. Experimental evaluations on a representative intrusion dataset demonstrate that our model enhances detection accuracy by 6.3%, reduces false positives by 9.0%, and maintains near real-time responsiveness. These results affirm the potential of language model integration to build intelligent, scalable, and resilient cybersecurity defences suited for modern connected environments.

摘要: 本文通过将传统的基于签名的方法与GPT-2大型语言模型（LLM）的上下文理解能力集成，提出了一种新颖的入侵检测方法。随着网络威胁变得越来越复杂，特别是在分布式、异类和资源受限的环境中，例如物联网（IoT）所支持的环境中，对动态和自适应入侵检测系统（IDS）的需求变得越来越紧迫。虽然传统方法对于检测已知威胁仍然有效，但它们通常无法识别新的和不断发展的攻击模式。相比之下，GPT-2擅长处理非结构化数据和识别复杂的语义关系，因此非常适合发现微妙的零日攻击载体。我们提出了一个混合IDS框架，该框架将基于签名的技术的稳健性与GPT-2驱动的语义分析的适应性相结合。对代表性入侵数据集的实验评估表明，我们的模型将检测准确性提高了6.3%，将假阳性降低了9.0%，并保持了近乎实时的响应能力。这些结果证实了语言模型集成在构建适合现代互联环境的智能、可扩展和弹性网络安全防御方面的潜力。



## **17. Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models**

Gen-AI时代的网络钓鱼检测：量化LLM与经典模型 cs.CR

8 Pages, IEEE Conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07406v1) [paper-pdf](http://arxiv.org/pdf/2507.07406v1)

**Authors**: Jikesh Thapa, Gurrehmat Chahal, Serban Voinea Gabreanu, Yazan Otoum

**Abstract**: Phishing attacks are becoming increasingly sophisticated, underscoring the need for detection systems that strike a balance between high accuracy and computational efficiency. This paper presents a comparative evaluation of traditional Machine Learning (ML), Deep Learning (DL), and quantized small-parameter Large Language Models (LLMs) for phishing detection. Through experiments on a curated dataset, we show that while LLMs currently underperform compared to ML and DL methods in terms of raw accuracy, they exhibit strong potential for identifying subtle, context-based phishing cues. We also investigate the impact of zero-shot and few-shot prompting strategies, revealing that LLM-rephrased emails can significantly degrade the performance of both ML and LLM-based detectors. Our benchmarking highlights that models like DeepSeek R1 Distill Qwen 14B (Q8_0) achieve competitive accuracy, above 80%, using only 17GB of VRAM, supporting their viability for cost-efficient deployment. We further assess the models' adversarial robustness and cost-performance tradeoffs, and demonstrate how lightweight LLMs can provide concise, interpretable explanations to support real-time decision-making. These findings position optimized LLMs as promising components in phishing defence systems and offer a path forward for integrating explainable, efficient AI into modern cybersecurity frameworks.

摘要: 网络钓鱼攻击变得越来越复杂，这凸显了对在高准确性和计算效率之间取得平衡的检测系统的需求。本文对传统机器学习（ML）、深度学习（DL）和用于网络钓鱼检测的量化小参数大型语言模型（LLM）进行了比较评估。通过对精心策划的数据集的实验，我们表明，虽然LLM目前在原始准确性方面表现不佳ML和DL方法，但它们在识别微妙的、基于上下文的网络钓鱼线索方面表现出强大的潜力。我们还研究了零激发和少激发策略的影响，揭示了LLM重新措辞的电子邮件会显着降低ML和基于LLM的检测器的性能。我们的基准测试强调，DeepSeek R1 Distill Qwen 14 B（Q8_0）等型号仅使用17 GB VRAM即可实现80%以上的竞争准确性，支持其具有成本效益的部署可行性。我们进一步评估了模型的对抗稳健性和成本-性能权衡，并展示了轻量级LLM如何提供简洁、可解释的解释以支持实时决策。这些发现将优化的LLM定位为网络钓鱼防御系统中有前途的组件，并为将可解释、高效的人工智能集成到现代网络安全框架中提供了前进的道路。



## **18. VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation**

Visual Trap：通过视觉基础操纵对图形用户界面代理进行秘密后门攻击 cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06899v1) [paper-pdf](http://arxiv.org/pdf/2507.06899v1)

**Authors**: Ziang Ye, Yang Zhang, Wentao Shi, Xiaoyu You, Fuli Feng, Tat-Seng Chua

**Abstract**: Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the visual grounding of GUI agent-mapping textual plans to GUI elements-can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent's behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose VisualTrap, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, e.g., being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents.

摘要: 由大型视觉语言模型（LVLM）驱动的图形用户界面（GUI）代理已经成为自动化人机交互的革命性方法，能够自主操作个人设备（例如，移动电话）或设备内的应用程序以类似于人类的方式执行复杂的现实世界任务。然而，它们与个人设备的紧密结合引发了重大的安全问题，包括后门攻击在内的许多威胁在很大程度上仍未得到解决。这项工作揭示了GUI代理的视觉基础-将文本计划映射到GUI元素-可以引入漏洞，从而实现新类型的后门攻击。通过针对视觉基础的后门攻击，即使给出了正确的任务解决计划，代理的行为也可能受到损害。为了验证此漏洞，我们提出了Visual Trap，这是一种可以通过误导代理定位文本计划来触发位置而不是预期目标来劫持接地的方法。Visual Trap使用注入有毒数据进行攻击的常见方法，并在视觉基础的预训练期间这样做，以确保攻击的实际可行性。经验结果表明，Visual Trap可以通过低至5%的有毒数据和高度隐蔽的视觉触发器（人眼看不见）有效劫持视觉基础;并且即使经过彻底的微调，攻击也可以推广到下游任务。此外，注入的触发器可以在不同的图形用户界面环境中保持有效，例如，正在接受移动/网络培训并推广到桌面环境。这些发现凸显了对图形用户界面代理后门攻击风险进行进一步研究的迫切性。



## **19. GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods**

GuidedBench：衡量和减轻野外LLM越狱方法的评估差异 cs.CL

Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2502.16903v2) [paper-pdf](http://arxiv.org/pdf/2502.16903v2)

**Authors**: Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Despite the growing interest in jailbreak methods as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. We conduct a systematic measurement study based on 37 jailbreak studies since 2022, focusing on both the methods and the evaluation systems they employ. We find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. This paper advocates a shift to a more nuanced, case-by-case evaluation paradigm. We introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset, detailed case-by-case evaluation guidelines and an evaluation system integrated with these guidelines -- GuidedEval. Experiments demonstrate that GuidedBench offers more accurate measurements of jailbreak performance, enabling meaningful comparisons across methods and uncovering new insights overlooked in previous evaluations. GuidedEval reduces inter-evaluator variance by at least 76.03\%. Furthermore, we observe that incorporating guidelines can enhance the effectiveness of jailbreak methods themselves, offering new insights into both attack strategies and evaluation paradigms.

摘要: 尽管人们越来越感兴趣越狱方法作为构建安全且负责任的大型语言模型（LLM）的有效红色团队工具，但有缺陷的评估系统设计导致其有效性评估存在显着差异。我们根据2022年以来的37项越狱研究进行了系统性的测量研究，重点关注他们采用的方法和评估体系。我们发现现有的评估系统缺乏针对具体案例的标准，导致对其有效性和安全性影响得出误导性结论。本文主张转向更加细致入微的逐个案例评估范式。我们引入了GuidedBench，这是一个新颖的基准，包括精心策划的有害问题数据集、详细的个案评估指南以及与这些指南集成的评估系统-- GuidedEval。实验表明，GuidedBench提供了更准确的越狱表现测量，能够进行各种方法之间有意义的比较，并发现之前评估中忽视的新见解。GuidedEval将评估者间方差减少至少76.03%。此外，我们观察到，纳入指南可以提高越狱方法本身的有效性，为攻击策略和评估范式提供新的见解。



## **20. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

尾部感知对抗攻击：高效LLM越狱的分布式方法 cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.04446v2) [paper-pdf](http://arxiv.org/pdf/2507.04446v2)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.

摘要: 为了保证大规模安全、稳健地部署大型语言模型（LLM），准确评估其对抗稳健性至关重要。现有的对抗性攻击通常针对单点贪婪世代的有害响应，忽视了LLM固有的随机性。在本文中，我们提出了一种新颖的对抗稳健性评估框架，该框架对整个输出分布（包括尾部风险）进行显式建模，为模型大规模稳健性提供更好的估计。通过将攻击过程描述为优化和采样之间的资源分配问题，我们确定了计算最优权衡，并表明将采样集成到现有攻击中可将ASB提高高达48%，并将效率提高高达两个数量级。我们的框架还使我们能够分析不同的攻击算法如何影响输出伤害分布。令人惊讶的是，我们发现大多数优化策略对输出危害影响很小。最后，我们引入了一个基于熵最大化的无数据概念验证目标，以演示我们的尾部感知视角如何实现新的优化目标。总体而言，我们的研究结果强调了尾部感知攻击和评估协议对于准确评估和加强LLM安全性的重要性。



## **21. An attention-aware GNN-based input defender against multi-turn jailbreak on LLMs**

一个具有注意力的基于GNN的输入防御者，防止LLM上的多回合越狱 cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.07146v1) [paper-pdf](http://arxiv.org/pdf/2507.07146v1)

**Authors**: Zixuan Huang, Kecheng Huang, Lihao Yin, Bowei He, Huiling Zhen, Mingxuan Yuan, Zili Shao

**Abstract**: Large Language Models (LLMs) have gained widespread popularity and are increasingly integrated into various applications. However, their capabilities can be exploited for both benign and harmful purposes. Despite rigorous training and fine-tuning for safety, LLMs remain vulnerable to jailbreak attacks. Recently, multi-turn attacks have emerged, exacerbating the issue. Unlike single-turn attacks, multi-turn attacks gradually escalate the dialogue, making them more difficult to detect and mitigate, even after they are identified.   In this study, we propose G-Guard, an innovative attention-aware GNN-based input classifier designed to defend against multi-turn jailbreak attacks on LLMs. G-Guard constructs an entity graph for multi-turn queries, explicitly capturing relationships between harmful keywords and queries even when those keywords appear only in previous queries. Additionally, we introduce an attention-aware augmentation mechanism that retrieves the most similar single-turn query based on the multi-turn conversation. This retrieved query is treated as a labeled node in the graph, enhancing the ability of GNN to classify whether the current query is harmful. Evaluation results demonstrate that G-Guard outperforms all baselines across all datasets and evaluation metrics.

摘要: 大型语言模型（LLM）已获得广泛流行，并越来越多地集成到各种应用程序中。然而，它们的能力可以被用于良性和有害的目的。尽管经过严格的培训和安全调整，LLM仍然容易受到越狱攻击。最近，出现了多回合攻击，加剧了这一问题。与单轮攻击不同，多轮攻击会逐渐升级对话，使其更难被发现和缓解，即使在被发现之后。   在这项研究中，我们提出了G-Guard，这是一种创新的基于注意力感知GNN的输入分类器，旨在抵御对LLM的多回合越狱攻击。G-Guard为多轮查询构建了一个实体图，显式地捕获有害关键字和查询之间的关系，即使这些关键字只出现在以前的查询中。此外，我们引入了一个注意力感知的增强机制，检索最相似的单轮查询的基础上的多轮对话。这个检索到的查询被视为图中的标签节点，增强了GNN对当前查询是否有害进行分类的能力。评估结果表明，G-Guard在所有数据集和评估指标中的表现优于所有基线。



## **22. Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions**

评估和改进大型语言模型的鲁棒性：调查和未来方向 cs.CL

33 pages, 5 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.11111v2) [paper-pdf](http://arxiv.org/pdf/2506.11111v2)

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.

摘要: 近年来，大型语言模型（LLM）因其理解和生成自然语言的能力而受到了广泛关注。随着快速发展和广泛应用（例如，代理人，联合情报），LLM的稳健性受到了越来越多的关注。作为许多人工智能应用的核心大脑，LLM的稳健性要求模型不仅要生成一致的内容，还要在处理意外的应用场景（例如，有毒提示、有限的噪音域数据、向外分布（OOD）应用程序等）。在这篇调查论文中，我们对LLM的稳健性进行了彻底的审查，旨在提供该领域的全面概念和方法术语并促进社区发展。具体来说，我们首先给出了LLM稳健性的正式定义，并给出了这篇调查论文的收集协议。然后，根据受干扰的输入类型，我们从以下角度组织本次调查：1）对抗稳健性：解决提示被故意操纵的问题，例如噪音提示、长上下文、数据攻击等; 2）OOD稳健性：处理意想不到的现实世界应用场景，例如OOD检测、零镜头传输、幻觉等; 3）稳健性评估：总结用于验证LLM稳健性的新评估数据集、指标和工具。在从各个角度回顾了代表性作品后，我们讨论并强调了该领域未来的机会和研究方向。同时，我们还组织相关工作并提供易于搜索的项目（https：//github.com/zhangkunzk/Awesome-LLM-Robustness-papers）来支持社区。



## **23. Breaking PEFT Limitations: Leveraging Weak-to-Strong Knowledge Transfer for Backdoor Attacks in LLMs**

打破PEFT限制：利用弱到强的知识转移进行LLM中的后门攻击 cs.CR

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2409.17946v4) [paper-pdf](http://arxiv.org/pdf/2409.17946v4)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Yanhao Jia, Luwei Xiao, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning (FPFT). However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from the weak-to-strong based on Feature Alignment-enhanced Knowledge Distillation (FAKD). Specifically, we poison small-scale language models through FPFT to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through FAKD, which employs PEFT. Theoretical analysis reveals that FAKD has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of FAKD on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.

摘要: 尽管大型语言模型（LLM）因其卓越的功能而被广泛应用，但已被证明容易受到后门攻击。这些攻击通过毒害训练样本和全参数微调（FPFT）将有针对性的漏洞引入LLM。然而，这种后门攻击是有限的，因为它们需要大量的计算资源，特别是随着LLM规模的增加。此外，参数高效微调（PEFT）提供了一种替代方案，但受限制的参数更新可能会阻碍触发器与目标标签的对齐。在这项研究中，我们首先验证了使用PEFT进行的后门攻击在实现可行的性能时可能会遇到挑战。为了解决这些问题并提高PEFT后门攻击的有效性，我们提出了一种基于特征对齐增强知识提炼（FAKD）的从弱到强的新型后门攻击算法。具体来说，我们通过FPFT毒害小规模语言模型，以充当教师模型。然后，教师模式通过采用PEFT的FAKD秘密地将后门转移到大规模学生模式。理论分析表明，FAKD有潜力增强后门攻击的有效性。我们展示了FAKD在四种语言模型、四种后门攻击算法和两种不同的教师模型架构上的分类任务上的卓越性能。实验结果表明，针对PEFT的后门攻击成功率接近100%。



## **24. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

21 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2411.08003v2) [paper-pdf](http://arxiv.org/pdf/2411.08003v2)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.

摘要: 在对抗环境中（例如网络攻击和虚假信息攻击）对大型语言模型（LLM）的输出进行归因会带来重大挑战，而且其重要性可能会越来越大。我们从理论和实证的角度来处理这个归因问题，借鉴形式语言理论（极限识别）和对不断扩大的LLM生态系统的数据驱动分析。通过将LLM的一组可能输出建模为形式语言，我们分析有限的文本样本是否可以唯一地确定原始模型。我们的结果表明，在模型之间能力重叠的温和假设下，某些类别的LLM从根本上无法仅从其输出中识别。我们描绘了理论可识别性的四种制度：（1）无限一类确定性（离散）LLM语言不可识别（Gold的经典结果来自1967年）;（2）无限类概率LLM也是不可识别的（通过确定性情况的扩展）;（3）有限类确定性LLM是可识别的（与Angluin的泄密标准一致）;以及（4）即使是有限类的概率LLM也可能是不可识别的（我们提供了一个新的反例来建立这个负结果）。作为对这些理论见解的补充，我们量化了近年来给定输出的合理模型起源（假设空间）数量的爆炸式增长。即使在保守的假设下--每个开源模型最多在一个新厕所上进行微调--不同候选模型的数量也大约每0.5年翻一番，并且允许多数据集微调组合可以产生翻倍的时间短至0.28年。这种组合增长，加上所有模型和潜在用户的暴力可能性归因的非凡计算成本，使得详尽的归因在实践中不可行。



## **25. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06489v1) [paper-pdf](http://arxiv.org/pdf/2507.06489v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于部署LLM至关重要，以确保许多高风险应用程序中人机交互的透明度、信任和安全。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们引入了一个新颖的框架，通过干扰和基于越狱的方法攻击言语信心分数，并表明这些攻击可能会显着危及言语信心估计并导致答案频繁变化。我们研究了各种提示策略、模型大小和应用领域，揭示了当前的信心激发方法很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了迫切需要设计更强大的机制来表达LLM的信心，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **26. Bridging AI and Software Security: A Comparative Vulnerability Assessment of LLM Agent Deployment Paradigms**

连接人工智能和软件安全：LLM代理部署范式的比较漏洞评估 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06323v1) [paper-pdf](http://arxiv.org/pdf/2507.06323v1)

**Authors**: Tarek Gasmi, Ramzi Guesmi, Ines Belhadj, Jihene Bennaceur

**Abstract**: Large Language Model (LLM) agents face security vulnerabilities spanning AI-specific and traditional software domains, yet current research addresses these separately. This study bridges this gap through comparative evaluation of Function Calling architecture and Model Context Protocol (MCP) deployment paradigms using a unified threat classification framework. We tested 3,250 attack scenarios across seven language models, evaluating simple, composed, and chained attacks targeting both AI-specific threats (prompt injection) and software vulnerabilities (JSON injection, denial-of-service). Function Calling showed higher overall attack success rates (73.5% vs 62.59% for MCP), with greater system-centric vulnerability while MCP exhibited increased LLM-centric exposure. Attack complexity dramatically amplified effectiveness, with chained attacks achieving 91-96% success rates. Counterintuitively, advanced reasoning models demonstrated higher exploitability despite better threat detection. Results demonstrate that architectural choices fundamentally reshape threat landscapes. This work establishes methodological foundations for cross-domain LLM agent security assessment and provides evidence-based guidance for secure deployment. Code and experimental materials are available at https: // github. com/ theconsciouslab-ai/llm-agent-security.

摘要: 大型语言模型（LLM）代理面临跨越人工智能特定和传统软件领域的安全漏洞，但当前的研究分别解决了这些问题。本研究通过使用统一的威胁分类框架对功能调用架构和模型上下文协议（HCP）部署范式进行比较评估来弥合这一差距。我们测试了七种语言模型中的3，250个攻击场景，评估了针对人工智能特定威胁（提示注入）和软件漏洞（SON注入、拒绝服务）的简单、组合和连锁攻击。功能调用显示出更高的总体攻击成功率（73.5% vs 62.59%），以系统为中心的脆弱性更大，而麦克唐纳则显示出以LLM为中心的暴露率增加。攻击的复杂性极大地提高了有效性，连锁攻击的成功率达到了91-96%。与直觉相反，尽管威胁检测更好，但高级推理模型仍表现出更高的可利用性。结果表明，架构选择从根本上重塑了威胁格局。这项工作为跨域LLM代理安全评估奠定了方法论基础，并为安全部署提供了基于证据的指导。代码和实验材料可在https：// github上获取。com/ theconsciouslab-ai/llm-Agent-secure。



## **27. CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations**

CAVGAN：通过对其内部代表的生成性对抗攻击统一LLM的越狱和辩护 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06043v1) [paper-pdf](http://arxiv.org/pdf/2507.06043v1)

**Authors**: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

**Abstract**: Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.

摘要: 安全对齐使大型语言模型（LLM）能够获得针对恶意查询的保护，但各种越狱攻击方法揭示了这种安全机制的漏洞。之前的研究已经孤立了LLM越狱攻击和防御。我们分析了LLM的安全保护机制，提出了攻击与防御相结合的框架。我们的方法基于LLM中间层嵌入的线性可分离性质，以及越狱攻击的本质，旨在嵌入有害问题并将其转移到安全区域。我们利用生成对抗网络（GAN）来学习LLM内部的安全判断边界，以实现高效的越狱攻击和防御。实验结果表明，我们的方法在三种流行的LLM中平均越狱成功率为88.85%，而在最先进的越狱数据集上的防御成功率平均达到84.17%。这不仅验证了我们方法的有效性，还揭示了LLM的内部安全机制，为增强模型安全性提供了新的见解。代码和数据可在https://github.com/NLPGM/CAVGAN上获取。



## **28. Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks**

增强LLM水印针对擦除和欺骗攻击的弹性 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06274v1) [paper-pdf](http://arxiv.org/pdf/2507.06274v1)

**Authors**: Huanming Shen, Baizhou Huang, Xiaojun Wan

**Abstract**: Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.

摘要: 水印是防止大型语言模型（LLM）滥用的一种有希望的防御方法，但它仍然容易受到擦洗和欺骗攻击。该漏洞源于由水印窗口大小决定的固有权衡：较小的窗口更难抵抗擦洗，但更容易进行反向工程，从而实现低成本的基于统计学的欺骗攻击。这项工作通过引入一种新颖的机制（等效纹理密钥）打破了这种权衡，其中水印窗口内的多个令牌可以独立支持检测。基于冗余度，我们提出了一种新的子词汇分解等效tExture密钥（SEEK）水印方案。它实现了帕累托改进，提高了针对擦除攻击的弹性，而不会损害欺骗的稳健性。实验证明了SEEK相对于现有方法的优越性，在不同的数据集设置中产生了+88.2%/+92.3%/+82.0%的欺骗鲁棒性收益，并产生了+10.2%/+6.4%/+24.6%的擦洗鲁棒性收益。



## **29. ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis**

ETrace：通过基于LLM的跟踪分析在智能合同中进行事件驱动的漏洞检测 cs.CR

4 pages, 1 figure. Submitted to the 16th Asia-Pacific Symposium on  Internetware (Internetware 2025)

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2506.15790v2) [paper-pdf](http://arxiv.org/pdf/2506.15790v2)

**Authors**: Chenyang Peng, Haijun Wang, Yin Wu, Hao Wu, Ming Fan, Yitao Zhao, Ting Liu

**Abstract**: With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.

摘要: 随着区块链技术在各个领域的深入应用，确保智能合约的安全性和稳定性已成为一项严峻的挑战。当前漏洞检测中的安全分析方法可以分为静态分析和动态分析方法。然而，这些现有的传统漏洞检测方法主要依赖于分析原始合同代码，并非所有智能合同都提供可访问代码。我们提出ETrace，一种新颖的事件驱动的智能合同漏洞检测框架，它通过LLM支持的跟踪分析来唯一地识别潜在漏洞，而无需访问源代码。通过从事务日志中提取细粒度事件序列，该框架利用大型语言模型（LLM）作为自适应语义解释器，通过思想链推理重建事件分析。ETrace实现模式匹配，以建立事务行为模式和已知攻击行为之间的因果联系。此外，我们通过初步实验结果验证了ETrace的有效性。



## **30. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

MEF：一个用于评估黑箱大语言模型脆弱性的能力感知多重加密框架 cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.23404v3) [paper-pdf](http://arxiv.org/pdf/2505.23404v3)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have revealed significant vulnerabilities in Large Language Models (LLMs), facilitating the evasion of alignment safeguards through increasingly sophisticated prompt manipulations. In this paper, we propose MEF, a capability-aware multi-encryption framework for evaluating vulnerabilities in black-box LLMs. Our key insight is that the effectiveness of jailbreak strategies can be significantly enhanced by tailoring them to the semantic comprehension capabilities of the target model. We present a typology that classifies LLMs into Type I and Type II based on their comprehension levels, and design adaptive attack strategies for each. MEF combines layered semantic mutations and dual-ended encryption techniques, enabling circumvention of input, inference, and output-level defenses. Experimental results demonstrate the superiority of our approach. Remarkably, it achieves a jailbreak success rate of 98.9\% on GPT-4o (29 May 2025 release). Our findings reveal vulnerabilities in current LLMs' alignment defenses.

摘要: 对抗性越狱攻击的最新进展揭示了大型语言模型（LLM）中的显着漏洞，通过日益复杂的提示操纵促进了对对齐保障措施的规避。在本文中，我们提出了MEF，这是一个用于评估黑匣子LLM中漏洞的功能感知多重加密框架。我们的主要见解是，通过根据目标模型的语义理解能力定制越狱策略，可以显着增强它们的有效性。我们提出了一种类型学，根据它们的理解水平将LLM分为I型和II型，并为每种类型设计自适应攻击策略。MEF结合了分层语义突变和双端加密技术，能够规避输入、推理和输出级防御。实验结果证明了我们方法的优越性。值得注意的是，它在GPT-4 o（2025年5月29日发布）上的越狱成功率达到了98.9%。我们的研究结果揭示了当前LLM对齐防御的漏洞。



## **31. Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs**

假动作和攻击：越狱和保护LLM的基于注意力的策略 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2410.16327v2) [paper-pdf](http://arxiv.org/pdf/2410.16327v2)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Zejian Chen, Litian Zhang, Zheng Liu, Lirong Qiu, Zaisheng Ye

**Abstract**: Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.

摘要: 越狱攻击可用于通过诱导大型语言模型（LLM）生成有害内容来访问大型语言模型（LLM）的漏洞。最常见的攻击方法是构建语义模糊的提示来混淆和误导LLM。为了访问安全性并揭示LLM输入提示和输出之间的内在关系，引入注意力权重的分布来分析潜在原因。通过统计分析方法，定义了一些新颖的指标来更好地描述注意力权重的分布，例如敏感词的注意力强度（Attn_SensWords）、基于注意力的上下文依赖分数（Attn_DepScore）和注意力分散量（Attn_Entropy）。利用这些指标的独特特征、射束搜索算法，并受军事策略“假动作攻击”的启发，提出了一种有效的越狱攻击策略--基于注意力的攻击（BA）。在ABA中，使用嵌套攻击提示来转移LLM的注意力分布。通过这种方式，可以使用输入中更无害的部分来吸引LLM的注意力。此外，在BA的推动下，还提出了一种有效的防御策略--基于注意力的防御（ABD）。与BA相比，ABD可以通过校准输入提示的注意力分布来增强LLM的鲁棒性。已经进行了一些比较实验来证明BA和ABD的有效性。因此，ABA和ABD都可以用于访问LLM的安全性。比较实验的结果也给出了一个逻辑解释，即注意权重的分配会对LLM的产出产生很大影响。



## **32. Circumventing Safety Alignment in Large Language Models Through Embedding Space Toxicity Attenuation**

通过嵌入空间毒性衰减来规避大型语言模型中的安全一致 cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.08020v1) [paper-pdf](http://arxiv.org/pdf/2507.08020v1)

**Authors**: Zhibo Zhang, Yuxi Li, Kailong Wang, Shuai Yuan, Ling Shi, Haoyu Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across domains such as healthcare, education, and cybersecurity. However, this openness also introduces significant security risks, particularly through embedding space poisoning, which is a subtle attack vector where adversaries manipulate the internal semantic representations of input data to bypass safety alignment mechanisms. While previous research has investigated universal perturbation methods, the dynamics of LLM safety alignment at the embedding level remain insufficiently understood. Consequently, more targeted and accurate adversarial perturbation techniques, which pose significant threats, have not been adequately studied.   In this work, we propose ETTA (Embedding Transformation Toxicity Attenuation), a novel framework that identifies and attenuates toxicity-sensitive dimensions in embedding space via linear transformations. ETTA bypasses model refusal behaviors while preserving linguistic coherence, without requiring model fine-tuning or access to training data. Evaluated on five representative open-source LLMs using the AdvBench benchmark, ETTA achieves a high average attack success rate of 88.61%, outperforming the best baseline by 11.34%, and generalizes to safety-enhanced models (e.g., 77.39% ASR on instruction-tuned defenses). These results highlight a critical vulnerability in current alignment strategies and underscore the need for embedding-aware defenses.

摘要: 大型语言模型（LLM）在医疗保健、教育和网络安全等领域取得了巨大的成功。然而，这种开放性也带来了重大的安全风险，特别是通过嵌入空间中毒，这是一种微妙的攻击向量，攻击者操纵输入数据的内部语义表示以绕过安全对齐机制。虽然以前的研究已经调查了通用的扰动方法，LLM安全对齐的动态嵌入水平仍然没有得到充分的理解。因此，更有针对性和准确的对抗扰动技术，构成重大威胁，尚未得到充分研究。   在这项工作中，我们提出了ETTA（嵌入转换毒性减弱），这是一个通过线性转换识别和减弱嵌入空间中毒性敏感维度的新型框架。ETTA绕过了模型拒绝行为，同时保持语言一致性，无需模型微调或访问训练数据。使用AdvBench基准对五个有代表性的开源LLM进行评估，ETTA实现了88.61%的高平均攻击成功率，比最佳基线高出11.34%，并推广到安全增强模型（例如，77.39%的ASCR在描述调整防御上）。这些结果凸显了当前对齐策略中的一个关键漏洞，并强调了对嵌入感知防御的必要性。



## **33. Disappearing Ink: Obfuscation Breaks N-gram Code Watermarks in Theory and Practice**

消失的墨水：模糊在理论和实践中破解了N-gram代码水印 cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05512v1) [paper-pdf](http://arxiv.org/pdf/2507.05512v1)

**Authors**: Gehao Zhang, Eugene Bagdasarian, Juan Zhai, Shiqing Ma

**Abstract**: Distinguishing AI-generated code from human-written code is becoming crucial for tasks such as authorship attribution, content tracking, and misuse detection. Based on this, N-gram-based watermarking schemes have emerged as prominent, which inject secret watermarks to be detected during the generation.   However, their robustness in code content remains insufficiently evaluated. Most claims rely solely on defenses against simple code transformations or code optimizations as a simulation of attack, creating a questionable sense of robustness. In contrast, more sophisticated schemes already exist in the software engineering world, e.g., code obfuscation, which significantly alters code while preserving functionality. Although obfuscation is commonly used to protect intellectual property or evade software scanners, the robustness of code watermarking techniques against such transformations remains largely unexplored.   In this work, we formally model the code obfuscation and prove the impossibility of N-gram-based watermarking's robustness with only one intuitive and experimentally verified assumption, distribution consistency, satisfied. Given the original false positive rate of the watermarking detection, the ratio that the detector failed on the watermarked code after obfuscation will increase to 1 - fpr.   The experiments have been performed on three SOTA watermarking schemes, two LLMs, two programming languages, four code benchmarks, and four obfuscators. Among them, all watermarking detectors show coin-flipping detection abilities on obfuscated codes (AUROC tightly surrounds 0.5). Among all models, watermarking schemes, and datasets, both programming languages own obfuscators that can achieve attack effects with no detection AUROC higher than 0.6 after the attack. Based on the theoretical and practical observations, we also proposed a potential path of robust code watermarking.

摘要: 区分人工智能生成的代码与人类编写的代码对于作者归属、内容跟踪和滥用检测等任务变得至关重要。基于此，基于N-gram的水印方案逐渐成为主流，它注入秘密水印以在生成过程中检测。   然而，它们在代码内容方面的稳健性仍然没有得到充分评估。大多数主张仅依赖于针对简单代码转换或代码优化的防御作为攻击模拟，从而产生了令人怀疑的稳健性。相比之下，软件工程领域已经存在更复杂的方案，例如，代码混淆，可以显着更改代码，同时保留功能。尽管混淆通常用于保护知识产权或逃避软件扫描器，但代码水印技术针对此类转换的鲁棒性在很大程度上仍未被探索。   在这项工作中，我们对代码混淆进行了正式建模，并证明了基于N-gram的水印的鲁棒性不可能，只需满足一个直观且经过实验验证的假设（分布一致性）即可满足。给定水印检测的原始假阳性率，检测器在混淆后对水印代码失败的比率将增加到1 - fpr。   实验已经进行了三个SOTA水印方案，两个LLM，两种编程语言，四个代码基准，和四个混淆器。其中，所有的水印检测器都表现出对混淆代码的硬币翻转检测能力（AUROC紧紧围绕0.5）。在所有的模型、水印方案和数据集中，两种编程语言都有混淆器，可以实现攻击后无检测AUROC高于0.6的攻击效果。在理论和实践观察的基础上，我们还提出了鲁棒代码水印的潜在路径。



## **34. Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models**

响应攻击：利用上下文启动来越狱大型语言模型 cs.CL

21 pages, 9 figures. Code and data available at  https://github.com/Dtc7w3PQ/Response-Attack

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05248v1) [paper-pdf](http://arxiv.org/pdf/2507.05248v1)

**Authors**: Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, Jing Shao

**Abstract**: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. Building on this insight, we propose Response Attack, which uses an auxiliary LLM to generate a mildly harmful response to a paraphrased version of the original malicious query. They are then formatted into the dialogue and followed by a succinct trigger prompt, thereby priming the target model to generate harmful content. Across eight open-source and proprietary LLMs, RA consistently outperforms seven state-of-the-art jailbreak techniques, achieving higher attack success rates. To mitigate this threat, we construct and release a context-aware safety fine-tuning dataset, which significantly reduces the attack success rate while preserving model capabilities. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack.

摘要: 上下文启动（早期的刺激会秘密地偏向后来的判断）为大型语言模型（LLM）提供了一个尚未探索的攻击表面。我们发现了一个上下文启动漏洞，其中对话中的先前响应可以将其后续行为引导到违反政策的内容上。基于这一见解，我们提出了响应攻击，它使用辅助LLM来对原始恶意查询的重述版本生成轻微有害的响应。然后将它们格式化为对话，然后是简洁的触发提示，从而启动目标模型以生成有害内容。在八种开源和专有LLM中，RA的性能始终优于七种最先进的越狱技术，实现了更高的攻击成功率。为了减轻这种威胁，我们构建并发布了一个上下文感知的安全微调数据集，这可以显着降低攻击成功率，同时保留模型功能。代码和数据可在https://github.com/Dtc7w3PQ/Response-Attack上获取。



## **35. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

坏与好的传输攻击：解释和增强多模式大型语言模型之间的对抗性传输 cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2405.20090v4) [paper-pdf](http://arxiv.org/pdf/2405.20090v4)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.

摘要: 多模式大型语言模型（MLLM）在跨模式交互中表现出出色的性能，但它们也存在对抗性漏洞。特别是，对抗性例子的可移植性仍然是一个持续的挑战。本文具体分析了MLLM之间对抗性转移性的表现，并确定了影响这一特征的关键因素。我们发现，MLLM的可移植性存在于具有相同视觉编码器的跨LLM场景中，并指出可能影响可移植性的\underline{\textit{两个关键因素}}。我们提供了两种语义级数据增强方法：添加图像补丁（AIP）和印刷增强可移植性方法（TATM），它们增强了对抗性示例跨MLLM的可移植性。为了探索对现实世界的潜在影响，我们利用了两项可能产生负面和积极社会影响的任务：\ding{182}有害内容插入和\ding{183}信息保护。



## **36. The Hidden Threat in Plain Text: Attacking RAG Data Loaders**

纯文本中的隐藏威胁：攻击RAG数据加载器 cs.CR

currently under submission

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05093v1) [paper-pdf](http://arxiv.org/pdf/2507.05093v1)

**Authors**: Alberto Castagnaro, Umberto Salviati, Mauro Conti, Luca Pajola, Simeone Pizzi

**Abstract**: Large Language Models (LLMs) have transformed human-machine interaction since ChatGPT's 2022 debut, with Retrieval-Augmented Generation (RAG) emerging as a key framework that enhances LLM outputs by integrating external knowledge. However, RAG's reliance on ingesting external documents introduces new vulnerabilities. This paper exposes a critical security gap at the data loading stage, where malicious actors can stealthily corrupt RAG pipelines by exploiting document ingestion.   We propose a taxonomy of 9 knowledge-based poisoning attacks and introduce two novel threat vectors -- Content Obfuscation and Content Injection -- targeting common formats (DOCX, HTML, PDF). Using an automated toolkit implementing 19 stealthy injection techniques, we test five popular data loaders, finding a 74.4% attack success rate across 357 scenarios. We further validate these threats on six end-to-end RAG systems -- including white-box pipelines and black-box services like NotebookLM and OpenAI Assistants -- demonstrating high success rates and critical vulnerabilities that bypass filters and silently compromise output integrity. Our results emphasize the urgent need to secure the document ingestion process in RAG systems against covert content manipulations.

摘要: 自ChatGPT 2022年首次亮相以来，大型语言模型（LLM）已经改变了人机交互，检索增强生成（RAG）成为通过集成外部知识增强LLM输出的关键框架。然而，RAG对吸收外部文档的依赖引入了新的漏洞。本文揭示了数据加载阶段的一个关键安全漏洞，恶意行为者可以通过利用文档摄入来悄悄破坏RAG管道。   我们提出了9种基于知识的中毒攻击的分类法，并引入了两种新型威胁载体--内容混淆和内容注入--针对常见格式（DOCX、HTML、PDF）。我们使用实施19种隐形注入技术的自动化工具包测试了五种流行的数据加载器，发现357个场景中的攻击成功率为74.4%。我们在六个端到端RAG系统上进一步验证了这些威胁，包括NotebookLM和OpenAI Assistant等白盒管道和黑匣子服务，展示了高成功率和绕过过滤器并悄悄损害输出完整性的关键漏洞。我们的结果强调迫切需要保护RAG系统中的文档摄入过程免受秘密内容操纵。



## **37. BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning**

BackFeed：一个高效且标准化的联邦学习后门攻击基准套件 cs.CR

Under review at NeurIPS'25

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04903v1) [paper-pdf](http://arxiv.org/pdf/2507.04903v1)

**Authors**: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) systems are vulnerable to backdoor attacks, where adversaries train their local models on poisoned data and submit poisoned model updates to compromise the global model. Despite numerous proposed attacks and defenses, divergent experimental settings, implementation errors, and unrealistic assumptions hinder fair comparisons and valid conclusions about their effectiveness in real-world scenarios. To address this, we introduce BackFed - a comprehensive benchmark suite designed to standardize, streamline, and reliably evaluate backdoor attacks and defenses in FL, with a focus on practical constraints. Our benchmark offers key advantages through its multi-processing implementation that significantly accelerates experimentation and the modular design that enables seamless integration of new methods via well-defined APIs. With a standardized evaluation pipeline, we envision BackFed as a plug-and-play environment for researchers to comprehensively and reliably evaluate new attacks and defenses. Using BackFed, we conduct large-scale studies of representative backdoor attacks and defenses across both Computer Vision and Natural Language Processing tasks with diverse model architectures and experimental settings. Our experiments critically assess the performance of proposed attacks and defenses, revealing unknown limitations and modes of failures under practical conditions. These empirical insights provide valuable guidance for the development of new methods and for enhancing the security of FL systems. Our framework is openly available at https://github.com/thinh-dao/BackFed.

摘要: 联邦学习（FL）系统很容易受到后门攻击，对手会根据有毒数据训练其本地模型并提交有毒模型更新以损害全局模型。尽管提出了许多攻击和防御，但不同的实验设置、实现错误和不切实际的假设阻碍了公平的比较和关于其在现实世界场景中有效性的有效性的有效结论。为了解决这个问题，我们引入了BackFed --一个全面的基准套件，旨在标准化、简化和可靠地评估FL中的后门攻击和防御，重点关注实际限制。我们的基准测试通过其多处理实施来提供关键优势，可以显着加速实验，并通过定义良好的API实现新方法的无缝集成。通过标准化的评估管道，我们将BackFeed设想为一个即插即用的环境，供研究人员全面可靠地评估新的攻击和防御。使用BackFeed，我们通过不同的模型架构和实验环境对计算机视觉和自然语言处理任务中的代表性后门攻击和防御进行了大规模研究。我们的实验批判性地评估了拟议攻击和防御的性能，揭示了实际条件下未知的限制和失败模式。这些经验见解为新方法的开发和增强FL系统的安全性提供了宝贵的指导。我们的框架可在https://github.com/thinh-dao/BackFed上公开获取。



## **38. Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems**

鼹鼠是谁？基于LLM的多Agent系统中意图隐藏恶意代理的建模和检测 cs.MA

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04724v1) [paper-pdf](http://arxiv.org/pdf/2507.04724v1)

**Authors**: Yizhe Xie, Congcong Zhu, Xinyue Zhang, Minghao Wang, Chi Liu, Minglu Zhu, Tianqing Zhu

**Abstract**: Multi-agent systems powered by Large Language Models (LLM-MAS) demonstrate remarkable capabilities in collaborative problem-solving. While LLM-MAS exhibit strong collaborative abilities, the security risks in their communication and coordination remain underexplored. We bridge this gap by systematically investigating intention-hiding threats in LLM-MAS, and design four representative attack paradigms that subtly disrupt task completion while maintaining high concealment. These attacks are evaluated in centralized, decentralized, and layered communication structures. Experiments conducted on six benchmark datasets, including MMLU, MMLU-Pro, HumanEval, GSM8K, arithmetic, and biographies, demonstrate that they exhibit strong disruptive capabilities. To identify these threats, we propose a psychology-based detection framework AgentXposed, which combines the HEXACO personality model with the Reid Technique, using progressive questionnaire inquiries and behavior-based monitoring. Experiments conducted on six types of attacks show that our detection framework effectively identifies all types of malicious behaviors. The detection rate for our intention-hiding attacks is slightly lower than that of the two baselines, Incorrect Fact Injection and Dark Traits Injection, demonstrating the effectiveness of intention concealment. Our findings reveal the structural and behavioral risks posed by intention-hiding attacks and offer valuable insights into securing LLM-based multi-agent systems through psychological perspectives, which contributes to a deeper understanding of multi-agent safety. The code and data are available at https://anonymous.4open.science/r/AgentXposed-F814.

摘要: 由大型语言模型（LLM-MAS）支持的多智能体系统在协作解决问题方面表现出了非凡的能力。虽然LLM-MAS表现出强大的协作能力，但其沟通和协调中的安全风险仍然没有得到充分的研究。我们通过系统性调查LLM-MAS中的意图隐藏威胁来弥合这一差距，并设计四种代表性的攻击范式，这些攻击范式微妙地扰乱任务完成，同时保持高度隐蔽性。这些攻击在集中式、分散式和分层的通信结构中进行评估。对六个基准数据集（包括MMLU、MMLU-Pro、HumanEval、GSM 8 K、算术和传记）进行的实验表明，它们表现出强大的破坏能力。为了识别这些威胁，我们提出了一个基于心理的检测框架AgentXposed，它将HEXACO性格模型与Reid技术相结合，使用渐进式问卷调查和基于行为的监控。对六种类型的攻击进行的实验表明，我们的检测框架可以有效识别所有类型的恶意行为。我们的意图隐藏攻击的检测率略低于两个基线，错误事实注入和黑暗特征注入，证明了意图隐藏的有效性。我们的研究结果揭示了意图隐藏攻击带来的结构和行为风险，并为通过心理学角度保护基于LLM的多智能体系统提供了宝贵的见解，这有助于更深入地理解多智能体安全性。代码和数据可在https://anonymous.4open.science/r/AgentXposed-F814上获取。



## **39. Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World**

攻击者的噪音可以在现实世界中操纵您的音频LLM cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.06256v1) [paper-pdf](http://arxiv.org/pdf/2507.06256v1)

**Authors**: Vinu Sankar Sadasivan, Soheil Feizi, Rajiv Mathews, Lun Wang

**Abstract**: This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures.

摘要: 本文研究了基于音频的大型语言模型（ALLM）（例如Qwen 2-Audio）的现实世界漏洞。我们首先证明对手可以精心设计隐秘的音频扰动来操纵ALLM表现出特定的有针对性的行为，例如引发对唤醒关键词的响应（例如，“嘿Qwen”），或触发有害行为（例如“更改我的日历事件”）。随后，我们表明，在用户与ALLM交互期间播放对抗性背景噪音会显着降低响应质量。至关重要的是，我们的研究说明了这些攻击对现实世界场景的可扩展性，当这些对抗性噪音通过空气播放时，会影响其他无辜用户。此外，我们还讨论了攻击的转移性以及潜在的防御措施。



## **40. Model Inversion Attacks on Llama 3: Extracting PII from Large Language Models**

对Lama 3的模型倒置攻击：从大型语言模型中提取PRI cs.LG

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04478v1) [paper-pdf](http://arxiv.org/pdf/2507.04478v1)

**Authors**: Sathesh P. Sivashanmugam

**Abstract**: Large language models (LLMs) have transformed natural language processing, but their ability to memorize training data poses significant privacy risks. This paper investigates model inversion attacks on the Llama 3.2 model, a multilingual LLM developed by Meta. By querying the model with carefully crafted prompts, we demonstrate the extraction of personally identifiable information (PII) such as passwords, email addresses, and account numbers. Our findings highlight the vulnerability of even smaller LLMs to privacy attacks and underscore the need for robust defenses. We discuss potential mitigation strategies, including differential privacy and data sanitization, and call for further research into privacy-preserving machine learning techniques.

摘要: 大型语言模型（LLM）已经改变了自然语言处理，但它们记忆训练数据的能力带来了巨大的隐私风险。本文研究了对Llama 3.2模型的模型倒置攻击，Llama 3.2模型是Meta开发的多语言LLM。通过使用精心设计的提示来查询模型，我们演示了如何提取个人可识别信息（PRI），例如密码、电子邮件地址和帐户号码。我们的研究结果强调了更小的LLM容易受到隐私攻击，并强调了强大防御的必要性。我们讨论了潜在的缓解策略，包括差异隐私和数据清理，并呼吁对保护隐私的机器学习技术进行进一步研究。



## **41. Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs**

注意力流失：对LLM越狱攻击和防御的机械理解 cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04365v1) [paper-pdf](http://arxiv.org/pdf/2507.04365v1)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment.

摘要: 随着大型语言模型（LLM）变得越来越重要，确保其安全性变得至关重要。越狱攻击利用漏洞绕过安全护栏，构成重大威胁。然而，导致这些攻击的机制还没有得到很好的了解。在本文中，我们揭示了越狱袭击期间发生的一种普遍现象：注意力流失。在这种现象期间，该模型在攻击过程中逐渐减少对用户查询中不安全请求的关注，最终导致越狱。我们表明，注意力滑动在各种越狱方法中是一致的，包括基于梯度的令牌替换、预算级模板细化和上下文学习。此外，我们评估了基于查询扰动的两种防御措施：Token Highlighter和SmoothLLM，发现它们间接缓解了注意力滑动，其有效性与所实现的缓解程度正相关。受这一发现的启发，我们提出了注意力尖锐化，这是一种新的防御方法，通过使用温度缩放来尖锐注意力分数分布来直接对抗注意力滑动。对四种领先的LLM（Gemma 2 - 9 B-It、Llama3.1-8B-It、Qwen 2.5 - 7 B-It、Mistral-7 B-It v0.2）的实验表明，我们的方法可以有效抵抗各种越狱攻击，同时保持AlpacaEval上良性任务的性能。重要的是，注意力尖锐不会引入额外的计算或内存负担，使其成为现实世界部署的高效实用解决方案。



## **42. Mass-Scale Analysis of In-the-Wild Conversations Reveals Complexity Bounds on LLM Jailbreaking**

野外对话的大规模分析揭示了法学硕士越狱的复杂性界限 cs.CL

Code: https://github.com/ACMCMC/risky-conversations Results:  https://huggingface.co/risky-conversations Visualizer:  https://huggingface.co/spaces/risky-conversations/Visualizer

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.08014v1) [paper-pdf](http://arxiv.org/pdf/2507.08014v1)

**Authors**: Aldan Creo, Raul Castro Fernandez, Manuel Cebrian

**Abstract**: As large language models (LLMs) become increasingly deployed, understanding the complexity and evolution of jailbreaking strategies is critical for AI safety.   We present a mass-scale empirical analysis of jailbreak complexity across over 2 million real-world conversations from diverse platforms, including dedicated jailbreaking communities and general-purpose chatbots. Using a range of complexity metrics spanning probabilistic measures, lexical diversity, compression ratios, and cognitive load indicators, we find that jailbreak attempts do not exhibit significantly higher complexity than normal conversations. This pattern holds consistently across specialized jailbreaking communities and general user populations, suggesting practical bounds on attack sophistication. Temporal analysis reveals that while user attack toxicity and complexity remains stable over time, assistant response toxicity has decreased, indicating improving safety mechanisms. The absence of power-law scaling in complexity distributions further points to natural limits on jailbreak development.   Our findings challenge the prevailing narrative of an escalating arms race between attackers and defenders, instead suggesting that LLM safety evolution is bounded by human ingenuity constraints while defensive measures continue advancing. Our results highlight critical information hazards in academic jailbreak disclosure, as sophisticated attacks exceeding current complexity baselines could disrupt the observed equilibrium and enable widespread harm before defensive adaptation.

摘要: 随着大型语言模型（LLM）的部署越来越多，了解越狱策略的复杂性和演变对于人工智能安全至关重要。   我们对来自不同平台（包括专用越狱社区和通用聊天机器人）的超过200万次现实世界对话中的越狱复杂性进行了大规模实证分析。使用涵盖概率测量、词汇多样性、压缩比和认知负荷指标的一系列复杂性指标，我们发现越狱尝试并没有表现出比正常对话明显更高的复杂性。这种模式在专业越狱社区和普通用户群体中始终存在，这表明攻击复杂性的实际界限。时间分析表明，虽然用户攻击毒性和复杂性随着时间的推移保持稳定，但辅助响应毒性有所下降，这表明安全机制正在改善。复杂性分布中缺乏乘势定律缩放进一步表明了越狱发展的自然限制。   我们的发现挑战了攻击者和防御者之间军备竞赛不断升级的流行说法，相反，表明LLM安全演变受到人类聪明才智限制的限制，而防御措施则继续推进。我们的结果强调了学术越狱披露中的关键信息风险，因为超过当前复杂性基线的复杂攻击可能会破坏观察到的平衡，并在防御适应之前造成广泛的伤害。



## **43. Hijacking JARVIS: Benchmarking Mobile GUI Agents against Unprivileged Third Parties**

劫持JARRIS：针对无特权第三方对移动图形用户界面代理进行基准测试 cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04227v1) [paper-pdf](http://arxiv.org/pdf/2507.04227v1)

**Authors**: Guohong Liu, Jialei Ye, Jiacheng Liu, Yuanchun Li, Wei Liu, Pengzhi Gao, Jian Luan, Yunxin Liu

**Abstract**: Mobile GUI agents are designed to autonomously execute diverse device-control tasks by interpreting and interacting with mobile screens. Despite notable advancements, their resilience in real-world scenarios where screen content may be partially manipulated by untrustworthy third parties remains largely unexplored. Owing to their black-box and autonomous nature, these agents are vulnerable to manipulations that could compromise user devices. In this work, we present the first systematic investigation into the vulnerabilities of mobile GUI agents. We introduce a scalable attack simulation framework AgentHazard, which enables flexible and targeted modifications of screen content within existing applications. Leveraging this framework, we develop a comprehensive benchmark suite comprising both a dynamic task execution environment and a static dataset of vision-language-action tuples, totaling over 3,000 attack scenarios. The dynamic environment encompasses 58 reproducible tasks in an emulator with various types of hazardous UI content, while the static dataset is constructed from 210 screenshots collected from 14 popular commercial apps. Importantly, our content modifications are designed to be feasible for unprivileged third parties. We evaluate 7 widely-used mobile GUI agents and 5 common backbone models using our benchmark. Our findings reveal that all examined agents are significantly influenced by misleading third-party content (with an average misleading rate of 28.8% in human-crafted attack scenarios) and that their vulnerabilities are closely linked to the employed perception modalities and backbone LLMs. Furthermore, we assess training-based mitigation strategies, highlighting both the challenges and opportunities for enhancing the robustness of mobile GUI agents. Our code and data will be released at https://agenthazard.github.io.

摘要: 移动图形用户界面代理旨在通过解释移动屏幕和与移动屏幕交互来自主执行各种设备控制任务。尽管取得了显着的进步，但它们在屏幕内容可能被不值得信赖的第三方部分操纵的现实世界场景中的弹性在很大程度上仍然没有被探索。由于它们的黑匣子和自治性质，这些代理很容易受到可能危及用户设备的操纵。在这项工作中，我们对移动图形用户界面代理的漏洞进行了首次系统性调查。我们引入了一个可扩展的攻击模拟框架AgentHazard，它可以灵活且有针对性地修改现有应用程序中的屏幕内容。利用这个框架，我们开发了一个全面的基准测试套件，其中包括动态任务执行环境和视觉-语言-动作二元组的静态数据集，总共超过3，000种攻击场景。动态环境包含具有各种类型危险UI内容的模拟器中的58个可重复任务，而静态数据集是根据从14个流行商业应用程序收集的210个屏幕截图构建的。重要的是，我们的内容修改旨在对无特权的第三方可行。我们评估7广泛使用的移动GUI代理和5个常见的骨干模型，使用我们的基准。我们的研究结果表明，所有受检查的代理都受到误导性第三方内容的显著影响（在人为攻击场景中，平均误导率为28.8%），并且他们的漏洞与所采用的感知模式和骨干LLM密切相关。此外，我们评估基于培训的缓解策略，突出的挑战和机遇，以提高移动GUI代理的鲁棒性。我们的代码和数据将在https://agenthazard.github.io上发布。



## **44. Can Large Language Models Automate the Refinement of Cellular Network Specifications?**

大型语言模型能否自动细化蜂窝网络规范？ cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04214v1) [paper-pdf](http://arxiv.org/pdf/2507.04214v1)

**Authors**: Jianshuo Dong, Tianyi Zhang, Feng Yan, Yuanjie Li, Hewu Li, Han Qiu

**Abstract**: Cellular networks serve billions of users globally, yet concerns about reliability and security persist due to weaknesses in 3GPP standards. However, traditional analysis methods, including manual inspection and automated tools, struggle with increasingly expanding cellular network specifications. This paper investigates the feasibility of Large Language Models (LLMs) for automated cellular network specification refinement. To advance it, we leverage 200,000+ approved 3GPP Change Requests (CRs) that document specification revisions, constructing a valuable dataset for domain tasks. We introduce CR-eval, a principled evaluation framework, and benchmark 16 state-of-the-art LLMs, demonstrating that top models can discover security-related weaknesses in over 127 out of 200 test cases within five trials. To bridge potential gaps, we explore LLM specialization techniques, including fine-tuning an 8B model to match or surpass advanced LLMs like GPT-4o and DeepSeek-R1. Evaluations on 30 cellular attacks identify open challenges for achieving full automation. These findings confirm that LLMs can automate the refinement of cellular network specifications and provide valuable insights to guide future research in this direction.

摘要: 蜂窝网络为全球数十亿用户提供服务，但由于3GPP标准的弱点，人们对可靠性和安全性的担忧仍然存在。然而，包括手动检查和自动化工具在内的传统分析方法难以应对日益扩大的蜂窝网络规范。本文研究了大型语言模型（LLM）用于自动蜂窝网络规范细化的可行性。为了推进这一进程，我们利用了200，000多个已批准的3GPP变更请求（CR），这些请求记录了规范修订，为领域任务构建了有价值的数据集。我们介绍了CR-eval，一个原则性的评估框架，并对16个最先进的LLM进行了基准测试，证明顶级模型可以在五次试验中发现200个测试用例中的127个与安全相关的弱点。为了弥合潜在的差距，我们探索LLM专业化技术，包括微调8B模型以匹配或超越GPT-4 o和DeepSeek-R1等高级LLM。对30种蜂窝攻击的评估确定了实现完全自动化的挑战。这些发现证实了LLM可以自动细化蜂窝网络规范，并提供有价值的见解，以指导未来在这一方向的研究。



## **45. False Alarms, Real Damage: Adversarial Attacks Using LLM-based Models on Text-based Cyber Threat Intelligence Systems**

虚假警报，真实损害：在基于文本的网络威胁情报系统上使用基于LLM的模型进行对抗性攻击 cs.CR

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.06252v1) [paper-pdf](http://arxiv.org/pdf/2507.06252v1)

**Authors**: Samaneh Shafee, Alysson Bessani, Pedro M. Ferreira

**Abstract**: Cyber Threat Intelligence (CTI) has emerged as a vital complementary approach that operates in the early phases of the cyber threat lifecycle. CTI involves collecting, processing, and analyzing threat data to provide a more accurate and rapid understanding of cyber threats. Due to the large volume of data, automation through Machine Learning (ML) and Natural Language Processing (NLP) models is essential for effective CTI extraction. These automated systems leverage Open Source Intelligence (OSINT) from sources like social networks, forums, and blogs to identify Indicators of Compromise (IoCs). Although prior research has focused on adversarial attacks on specific ML models, this study expands the scope by investigating vulnerabilities within various components of the entire CTI pipeline and their susceptibility to adversarial attacks. These vulnerabilities arise because they ingest textual inputs from various open sources, including real and potentially fake content. We analyse three types of attacks against CTI pipelines, including evasion, flooding, and poisoning, and assess their impact on the system's information selection capabilities. Specifically, on fake text generation, the work demonstrates how adversarial text generation techniques can create fake cybersecurity and cybersecurity-like text that misleads classifiers, degrades performance, and disrupts system functionality. The focus is primarily on the evasion attack, as it precedes and enables flooding and poisoning attacks within the CTI pipeline.

摘要: 网络威胁情报（RTI）已成为一种重要的补充方法，在网络威胁生命周期的早期阶段运作。RTI涉及收集、处理和分析威胁数据，以更准确、更快速地了解网络威胁。由于数据量大，通过机器学习（ML）和自然语言处理（NLP）模型实现自动化对于有效的RTI提取至关重要。这些自动化系统利用来自社交网络、论坛和博客等来源的开源情报（Osint）来识别妥协指标（IoCs）。尽管之前的研究重点是对特定ML模型的对抗攻击，但这项研究通过调查整个RTI管道的各个组件内的漏洞及其对对抗攻击的易感性来扩大了范围。这些漏洞的出现是因为它们从各种开源获取文本输入，包括真实和潜在虚假内容。我们分析了针对RTI管道的三种类型的攻击，包括规避、洪水和中毒，并评估它们对系统信息选择能力的影响。具体来说，在虚假文本生成方面，该工作展示了对抗性文本生成技术如何创建虚假网络安全和类似网络安全的文本，从而误导分类器、降低性能并扰乱系统功能。重点主要是规避攻击，因为它先于RTI管道内的洪水和中毒攻击。



## **46. Membership Inference Attacks on Large-Scale Models: A Survey**

对大规模模型的成员推断攻击：一项调查 cs.LG

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2503.19338v2) [paper-pdf](http://arxiv.org/pdf/2503.19338v2)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: The adoption of the Large Language Model (LLM) has accelerated dramatically since ChatGPT from OpenAI went online in November 2022. Recent advances in Large Multimodal Models (LMMs), which process diverse data types and enable interaction through various channels, have expanded beyond the text-to-text limitations of early LLMs, attracting significant and concurrent attention from both researchers and industry. While LLMs and LMMs are starting to spread widely, concerns about their privacy risks are increasing as well. Membership Inference Attacks (MIAs) are techniques used to determine whether a particular data point was part of a model's training set, which is a key metric for assessing the privacy vulnerabilities of machine learning models. Hu et al. show that various machine learning algorithms are vulnerable to MIA. Despite extensive studies on MIAs in classic models, there remains a lack of systematic surveys addressing their effectiveness and limitations in advanced large-scale models like LLMs and LMMs. In this paper, we systematically reviewed recent studies of MIA against LLMs and LMMs. We analyzed and categorized each attack based on its methodology, scenario, and targeted model, and we discussed the limitations of existing research. In addition to examining attacks on pre-training and fine-tuning stages, we also explore MIAs that target other development pipelines, including Retrieval-Augmented Generation (RAG) and the model alignment process. Based on the survey, we provide suggestions for future studies to improve the robustness of MIA in large-scale AI models.

摘要: 自OpenAI的ChatGPT于2022年11月上线以来，大型语言模型（LLM）的采用急剧加速。大型多模式模型（LSYS）的最新进展处理不同的数据类型并通过各种渠道实现交互，已经超越了早期LLM的文本到文本限制，吸引了研究人员和行业的高度关注。虽然LLM和LSYS开始广泛传播，但对其隐私风险的担忧也在增加。成员资格推理攻击（MIA）是用于确定特定数据点是否是模型训练集的一部分的技术，这是评估机器学习模型隐私漏洞的关键指标。Hu等人表明，各种机器学习算法都容易受到MIA的影响。尽管对经典模型中的MIA进行了广泛的研究，但仍然缺乏系统性的调查来解决它们在LLM和LSYS等先进大规模模型中的有效性和局限性。在本文中，我们系统地回顾了最近关于MIA对抗LLM和LSYS的研究。我们根据方法论、场景和目标模型分析和分类了每种攻击，并讨论了现有研究的局限性。除了检查对预训练和微调阶段的攻击外，我们还探索针对其他开发管道的MIA，包括检索增强生成（RAG）和模型对齐过程。根据调查，我们为未来的研究提供建议，以提高MIA在大规模人工智能模型中的稳健性。



## **47. A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models**

大型语言模型中针对错误信息的主动防御策略研究 cs.IR

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.05288v1) [paper-pdf](http://arxiv.org/pdf/2507.05288v1)

**Authors**: Shuliang Liu, Hongyi Liu, Aiwei Liu, Bingchen Duan, Qi Zheng, Yibo Yan, He Geng, Peijie Jiang, Jia Liu, Xuming Hu

**Abstract**: The widespread deployment of large language models (LLMs) across critical domains has amplified the societal risks posed by algorithmically generated misinformation. Unlike traditional false content, LLM-generated misinformation can be self-reinforcing, highly plausible, and capable of rapid propagation across multiple languages, which traditional detection methods fail to mitigate effectively. This paper introduces a proactive defense paradigm, shifting from passive post hoc detection to anticipatory mitigation strategies. We propose a Three Pillars framework: (1) Knowledge Credibility, fortifying the integrity of training and deployed data; (2) Inference Reliability, embedding self-corrective mechanisms during reasoning; and (3) Input Robustness, enhancing the resilience of model interfaces against adversarial attacks. Through a comprehensive survey of existing techniques and a comparative meta-analysis, we demonstrate that proactive defense strategies offer up to 63\% improvement over conventional methods in misinformation prevention, despite non-trivial computational overhead and generalization challenges. We argue that future research should focus on co-designing robust knowledge foundations, reasoning certification, and attack-resistant interfaces to ensure LLMs can effectively counter misinformation across varied domains.

摘要: 大型语言模型（LLM）在关键领域的广泛部署放大了算法生成的错误信息带来的社会风险。与传统的虚假内容不同，LLM生成的错误信息可以自我强化、高度可信，并且能够在多种语言中快速传播，而传统检测方法无法有效缓解这一点。本文引入了一种主动防御范式，从被动事后检测转向预期缓解策略。我们提出了一个三柱框架：（1）知识可信度，加强训练和部署数据的完整性;（2）推理可靠性，在推理过程中嵌入自我纠正机制;（3）输入鲁棒性，增强模型接口针对对抗性攻击的弹性。通过对现有技术的全面调查和比较荟萃分析，我们证明，尽管存在重要的计算费用和概括性挑战，但主动防御策略在错误信息预防方面比传统方法提供了高达63%的改进。我们认为，未来的研究应该重点关注共同设计强大的知识基础、推理认证和抗攻击界面，以确保LLM能够有效地对抗各个领域的错误信息。



## **48. We Urgently Need Privilege Management in MCP: A Measurement of API Usage in MCP Ecosystems**

我们迫切需要MCP中的植物管理：MCP生态系统中API使用的测量 cs.CR

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.06250v1) [paper-pdf](http://arxiv.org/pdf/2507.06250v1)

**Authors**: Zhihao Li, Kun Li, Boyang Ma, Minghui Xu, Yue Zhang, Xiuzhen Cheng

**Abstract**: The Model Context Protocol (MCP) has emerged as a widely adopted mechanism for connecting large language models to external tools and resources. While MCP promises seamless extensibility and rich integrations, it also introduces a substantially expanded attack surface: any plugin can inherit broad system privileges with minimal isolation or oversight. In this work, we conduct the first large-scale empirical analysis of MCP security risks. We develop an automated static analysis framework and systematically examine 2,562 real-world MCP applications spanning 23 functional categories. Our measurements reveal that network and system resource APIs dominate usage patterns, affecting 1,438 and 1,237 servers respectively, while file and memory resources are less frequent but still significant. We find that Developer Tools and API Development plugins are the most API-intensive, and that less popular plugins often contain disproportionately high-risk operations. Through concrete case studies, we demonstrate how insufficient privilege separation enables privilege escalation, misinformation propagation, and data tampering. Based on these findings, we propose a detailed taxonomy of MCP resource access, quantify security-relevant API usage, and identify open challenges for building safer MCP ecosystems, including dynamic permission models and automated trust assessment.

摘要: 模型上下文协议（HCP）已成为一种广泛采用的将大型语言模型连接到外部工具和资源的机制。虽然HCP承诺无缝的可扩展性和丰富的集成，但它也引入了大幅扩展的攻击面：任何插件都可以在最小的隔离或监督的情况下继承广泛的系统特权。在这项工作中，我们对LCP安全风险进行了首次大规模实证分析。我们开发了一个自动化静态分析框架，并系统性地检查了涵盖23个功能类别的2，562个现实世界的LCP应用程序。我们的测量显示，网络和系统资源API主导了使用模式，分别影响1，438和1，237台服务器，而文件和内存资源的频率较低，但仍然很重要。我们发现开发人员工具和API开发插件是API最密集的，而不太受欢迎的插件通常包含不成比例的高风险操作。通过具体的案例研究，我们展示了权限分离不足如何导致权限升级、错误信息传播和数据篡改。基于这些研究结果，我们提出了一个详细的LCP资源访问分类法，量化与安全相关的API使用情况，并确定构建更安全的LCP生态系统的公开挑战，包括动态许可模型和自动信任评估。



## **49. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

探索LLM中的潜在子空间以实现人工智能安全：识别和操纵敌对状态 cs.LG

4 figures

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2503.09066v2) [paper-pdf](http://arxiv.org/pdf/2503.09066v2)

**Authors**: Xin Wei Chia, Swee Liang Wong, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的能力，但它们仍然容易受到对抗操纵的影响，例如通过提示注入攻击进行越狱。这些攻击绕过安全机制来生成受限制或有害内容。在这项研究中，我们通过从LLM中提取隐藏激活来研究安全和越狱状态的潜在子空间。受神经科学中吸引子动力学的启发，我们假设LLM激活会进入半稳定状态，可以识别和扰动这些状态以引发状态转变。使用降维技术，我们预测安全和越狱反应的激活，以揭示低维空间中的潜在子空间。然后，我们推导出一个扰动载体，当将其应用于安全表示时，会将模型转向越狱状态。我们的结果表明，这种因果干预会在提示子集中导致具有统计学意义的越狱反应。接下来，我们探讨了这些扰动如何在模型的层中传播，测试诱导的状态变化是保持局部化还是在整个网络中级联。我们的研究结果表明，有针对性的扰动会导致激活和模型响应的明显变化。我们的方法为潜在的主动防御铺平了道路，从传统的基于护栏的方法转向先发制人的、模型不可知的技术，可以在表示层面中和对抗状态。



## **50. Blackbox Dataset Inference for LLM**

LLM的黑匣子数据集推理 cs.CR

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03619v1) [paper-pdf](http://arxiv.org/pdf/2507.03619v1)

**Authors**: Ruikai Zhou, Kang Yang, Xun Chen, Wendy Hui Wang, Guanhong Tao, Jun Xu

**Abstract**: Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs.   In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.

摘要: 如今，大型语言模型（LLM）的训练可能涉及个人可识别信息和受版权保护的材料，从而导致数据集滥用。为了缓解数据集滥用的问题，本文探讨了\textit{dataset initiation}，其目的是检测可疑模型$\mathCal{M}$是否在训练中使用了受害者数据集$\mathCal{D}$。之前的研究通过聚集隶属度推理攻击（MIA）的结果来解决数据集推理--MIA是确定单个样本是否是训练数据集一部分的方法。然而，受MIA准确性低的限制，之前的研究要求灰箱访问$\mathCal{M}$以获得中间输出（概率、损失、困惑度等）以获得满意的结果。这导致实用性降低，因为LLM，尤其是那些为利润而部署的LLM，返回中间产出的动力有限。   在本文中，我们提出了一种新的数据集推理方法，仅通过黑匣子访问目标模型（即，假设只有目标模型的基于文本的响应可用）。我们的方法由两组本地构建的参考模型来支持，一组在训练中涉及$\mathCal{D}$，另一组不涉及。通过测量$\mathCal{M}$更接近哪一组参考模型，我们确定$\mathCal{M}$是否使用$\mathCal{D}$进行训练。对现实世界LLM的野外评估表明，我们的方法在所有设置中都提供了高准确性，并且具有针对绕过尝试的鲁棒性。



