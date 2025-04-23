# Latest Large Language Model Attack Papers
**update at 2025-04-23 10:03:56**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多模式大型语言模型（MLLM）在各种多模式任务中展示了令人印象深刻的性能。另一方面，额外图像形态的集成可能会允许恶意用户在图像中注入有害内容以进行越狱。与基于文本的LLM不同，对手需要选择离散令牌来使用特定算法隐藏其恶意意图，图像信号的连续性为对手提供了注入有害意图的直接机会。在这项工作中，我们提出了$\textBF{BaThe}$（$\textBF{BA}$ckdoor $\textBF{T}$rigger S$\textBF{h}$i$\textBF{e}$ld），这是一种简单而有效的越狱防御机制。我们的工作受到最近对生成性语言模型中越狱后门攻击和虚拟提示后门攻击的研究的启发。越狱后门攻击使用有害指令与手工制作的字符串相结合作为触发器，使后门模型生成禁止的响应。我们假设有害指令可以充当触发器，如果我们将拒绝响应设置为触发响应，那么后门模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一目标，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为“wedge”。我们的全面实验表明，BaThe有效地缓解了各种类型的越狱攻击，并且能够抵御不可见的攻击，对MLLM的性能影响最小。



## **2. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

Red Team Diffuser：通过强化学习暴露视觉语言模型中的有毒连续漏洞 cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.

摘要: 大型视觉语言模型（VLM）的不断增加的部署暴露了其对齐机制中的关键安全漏洞。虽然现有的越狱研究主要关注VLM对有害指令的敏感性，但我们揭示了一个基本但被忽视的弱点：有毒文本延续，当提示有害文本前置与语义对抗图像配对时，VLM会产生剧毒的完成。为了系统性地研究这种威胁，我们提出了Red Team Distuser（RTI），这是第一个红色团队扩散模型，通过强化学习协调对抗图像生成和有毒延续。我们的关键创新包括动态跨模式攻击和隐身优化。对于LLM安全基准中的有毒文本前置，我们进行贪婪搜索以识别最大限度地引发有毒完成的最佳图像提示。发现的图像提示然后驱动基于RL的扩散模型微调，产生语义对齐的对抗图像，从而提高毒性率。潜行感知优化引入了联合对抗奖励，平衡毒性最大化（通过Dealfy分类器）和潜行性（通过BERTScore），规避了传统的基于噪音的对抗模式。实验结果证明了RTI的有效性，在原始攻击集中，LLaVA输出的毒性率比纯文本基线增加了10.69%，在未见集上增加了8.91%，证明了概括能力。此外，RTI具有较强的跨模型转移性，使Gemini的毒性率提高了5.1%，对LLaMA的毒性率提高了26.83%。我们的研究结果暴露了当前VLM对齐中的两个关键缺陷：（1）未能防止有害前置的有毒延续，以及（2）忽视了跨模式攻击载体。这些结果需要在安全性评估中向多模式红色团队转变。



## **3. Exploring the Role of Large Language Models in Cybersecurity: A Systematic Survey**

探索大型语言模型在网络安全中的作用：系统性调查 cs.CR

20 pages, 3 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15622v1) [paper-pdf](http://arxiv.org/pdf/2504.15622v1)

**Authors**: Shuang Tian, Tao Zhang, Jiqiang Liu, Jiacheng Wang, Xuangou Wu, Xiaoqiang Zhu, Ruichen Zhang, Weiting Zhang, Zhenhui Yuan, Shiwen Mao, Dong In Kim

**Abstract**: With the rapid development of technology and the acceleration of digitalisation, the frequency and complexity of cyber security threats are increasing. Traditional cybersecurity approaches, often based on static rules and predefined scenarios, are struggling to adapt to the rapidly evolving nature of modern cyberattacks. There is an urgent need for more adaptive and intelligent defence strategies. The emergence of Large Language Model (LLM) provides an innovative solution to cope with the increasingly severe cyber threats, and its potential in analysing complex attack patterns, predicting threats and assisting real-time response has attracted a lot of attention in the field of cybersecurity, and exploring how to effectively use LLM to defend against cyberattacks has become a hot topic in the current research field. This survey examines the applications of LLM from the perspective of the cyber attack lifecycle, focusing on the three phases of defense reconnaissance, foothold establishment, and lateral movement, and it analyzes the potential of LLMs in Cyber Threat Intelligence (CTI) tasks. Meanwhile, we investigate how LLM-based security solutions are deployed and applied in different network scenarios. It also summarizes the internal and external risk issues faced by LLM during its application. Finally, this survey also points out the facing risk issues and possible future research directions in this domain.

摘要: 随着技术的快速发展和数字化进程的加快，网络安全威胁的频率和复杂性不断增加。传统的网络安全方法通常基于静态规则和预定义的场景，正在努力适应现代网络攻击快速变化的性质。迫切需要更具适应性和智能性的防御策略。大型语言模型（LLM）的出现为应对日益严重的网络威胁提供了创新解决方案，其在分析复杂攻击模式、预测威胁和辅助实时响应方面的潜力引起了网络安全领域的广泛关注，探索如何有效利用LLM防御网络攻击已成为当前研究领域的热门话题。本次调查从网络攻击生命周期的角度审视了LLM的应用，重点关注防御侦察、立足点建立和侧向移动三个阶段，并分析了LLM在网络威胁情报（RTI）任务中的潜力。同时，我们研究了基于LLM的安全解决方案如何在不同的网络场景中部署和应用。还总结了LLM在应用过程中面临的内部和外部风险问题。最后，本次调查还指出了该领域面临的风险问题以及未来可能的研究方向。



## **4. Diversity Helps Jailbreak Large Language Models**

多样性帮助越狱大型语言模型 cs.CL

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2411.04223v2) [paper-pdf](http://arxiv.org/pdf/2411.04223v2)

**Authors**: Weiliang Zhao, Daniel Ben-Levi, Wei Hao, Junfeng Yang, Chengzhi Mao

**Abstract**: We have uncovered a powerful jailbreak technique that leverages large language models' ability to diverge from prior context, enabling them to bypass safety constraints and generate harmful outputs. By simply instructing the LLM to deviate and obfuscate previous attacks, our method dramatically outperforms existing approaches, achieving up to a 62.83% higher success rate in compromising ten leading chatbots, including GPT-4, Gemini, and Llama, while using only 12.9% of the queries. This revelation exposes a critical flaw in current LLM safety training, suggesting that existing methods may merely mask vulnerabilities rather than eliminate them. Our findings sound an urgent alarm for the need to revolutionize testing methodologies to ensure robust and reliable LLM security.

摘要: 我们发现了一种强大的越狱技术，该技术利用大型语言模型脱离先前上下文的能力，使它们能够绕过安全约束并生成有害输出。通过简单地指示LLM偏离和混淆之前的攻击，我们的方法显着优于现有方法，在攻击包括GPT-4、Gemini和Llama在内的十个领先聊天机器人时，成功率提高了62.83%，而仅使用12.9%的查询。这一揭露暴露了当前LLM安全培训中的一个关键缺陷，表明现有方法可能只是掩盖了漏洞而不是消除漏洞。我们的发现敲响了紧急警报，需要彻底改变测试方法，以确保强大和可靠的LLM安全。



## **5. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

通过跨模式提示注射操纵多模式代理 cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.14348v2) [paper-pdf](http://arxiv.org/pdf/2504.14348v2)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.

摘要: 多模式大型语言模型的出现通过将语言和视觉模式与外部数据源集成来重新定义了代理范式，使代理能够更好地解释人类指令并执行日益复杂的任务。然而，在这项工作中，我们发现了多模式代理中一个以前被忽视的关键安全漏洞：跨模式提示注入攻击。为了利用这个漏洞，我们提出了CrossInib，这是一种新型攻击框架，其中攻击者在多种模式中嵌入对抗性扰动，以与目标恶意内容保持一致，允许外部指令劫持代理的决策过程并执行未经授权的任务。我们的方法由两个关键部分组成。首先，我们引入了视觉潜在对齐，基于文本到图像生成模型，优化视觉嵌入空间中恶意指令的对抗特征，确保对抗图像巧妙地编码恶意任务执行的线索。随后，我们提出了文本指导增强，其中利用大型语言模型通过对抗性Meta提示来推断黑匣子防御系统提示，并生成恶意文本命令，该命令引导代理的输出更好地遵守攻击者的请求。大量实验表明，我们的方法优于现有的注入攻击，在不同任务中的攻击成功率至少增加了+26.4%。此外，我们还验证了攻击在现实世界的多模式自治代理中的有效性，强调了其对安全关键应用程序的潜在影响。



## **6. Trading Devil RL: Backdoor attack via Stock market, Bayesian Optimization and Reinforcement Learning**

交易魔鬼RL：通过股市、Bayesian优化和强化学习进行后门攻击 cs.LG

End of data poisoning research!: Navier-stokes equations (3D;  update); Reinforcement Learning (RL); HFT (High Frequency Trading); Limit  Order Markets and backdoor attack detection

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2412.17908v3) [paper-pdf](http://arxiv.org/pdf/2412.17908v3)

**Authors**: Orson Mengara

**Abstract**: With the rapid development of generative artificial intelligence, particularly large language models a number of sub-fields of deep learning have made significant progress and are now very useful in everyday applications. For example,financial institutions simulate a wide range of scenarios for various models created by their research teams using reinforcement learning, both before production and after regular operations. In this work, we propose a backdoor attack that focuses solely on data poisoning and a method of detection by dynamic systems and statistical analysis of the distribution of data. This particular backdoor attack is classified as an attack without prior consideration or trigger, and we name it FinanceLLMsBackRL. Our aim is to examine the potential effects of large language models that use reinforcement learning systems for text production or speech recognition, finance, physics, or the ecosystem of contemporary artificial intelligence models.

摘要: 随着生成式人工智能，特别是大型语言模型的快速发展，深度学习的许多子领域取得了重大进展，现在在日常应用中非常有用。例如，金融机构在生产之前和常规运营之后使用强化学习为其研究团队创建的各种模型模拟各种场景。在这项工作中，我们提出了一种仅关注数据中毒的后门攻击，以及一种通过动态系统和数据分布统计分析进行检测的方法。这种特殊的后门攻击被归类为未经事先考虑或触发的攻击，我们将其命名为Financial LLMsBackRL。我们的目标是研究使用强化学习系统的大型语言模型对文本生成或语音识别、金融、物理或当代人工智能模型生态系统的潜在影响。



## **7. LAMD: Context-driven Android Malware Detection and Classification with LLMs**

LAMD：使用LLM的上下文驱动Android恶意软件检测和分类 cs.CR

accepted by 2025 46th IEEE Symposium on Security and Privacy  Workshops (SPW)

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.13055v2) [paper-pdf](http://arxiv.org/pdf/2502.13055v2)

**Authors**: Xingzhi Qian, Xinran Zheng, Yiling He, Shuo Yang, Lorenzo Cavallaro

**Abstract**: The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes.

摘要: 移动应用程序的快速增长加剧了Android恶意软件威胁。尽管有多种检测方法，但它们经常与不断发展的攻击、数据集偏差和有限的解释性作斗争。大型语言模型（LLM）凭借其零触发推理和推理能力提供了一个有前途的替代方案。然而，将LLM应用于Android恶意软件检测存在两个关键挑战：（1）Android应用程序中的广泛支持代码（通常跨越数千个类）超出了LLM的上下文限制，并掩盖了良性功能内的恶意行为;（2）Android应用程序的结构复杂性和相互依赖性超过了LLM基于序列的推理、碎片化代码分析并阻碍恶意意图推断。为了应对这些挑战，我们提出了LAMD，这是一个实用的上下文驱动框架，用于支持基于LLM的Android恶意软件检测。LAMD集成关键上下文提取来隔离安全关键代码区域并构建程序结构，然后应用分层代码推理来逐步分析应用程序行为，从低级指令到高级语义，提供最终的预测和解释。设计良好的事实一致性验证机制可以从第一层减轻LLM幻觉。现实环境中的评估证明了LAMD相对于传统检测器的有效性，为动态威胁环境中LLM驱动的恶意软件分析奠定了可行的基础。



## **8. ASIDE: Architectural Separation of Instructions and Data in Language Models**

ASIDE：语言模型中指令和数据的架构分离 cs.LG

ICLR 2025 Workshop on Building Trust in Language Models and  Applications

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.10566v2) [paper-pdf](http://arxiv.org/pdf/2503.10566v2)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Alexandra Volkova, Soroush Tabesh, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, and this makes them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause for the success of prompt injection attacks. In this work, we propose a method, ASIDE, that allows the model to clearly separate between instructions and data on the level of embeddings. ASIDE applies a fixed orthogonal rotation to the embeddings of data tokens, thus creating distinct representations of instructions and data tokens without introducing any additional parameters. We demonstrate the effectiveness of our method by instruct-tuning LLMs with ASIDE and showing (1) highly increased instruction-data separation scores without a loss in model capabilities and (2) competitive results on prompt injection benchmarks, even without dedicated safety training. Additionally, we study the working mechanism behind our method through an analysis of model representations.

摘要: 尽管大型语言模型性能出色，但缺乏基本的安全功能，这使得它们容易受到大量恶意攻击。特别是，之前的工作已经确定指令和数据之间缺乏内在分离是提示注入攻击成功的根本原因。在这项工作中，我们提出了一种方法ASIDE，该方法允许模型在嵌入级别上清楚地分离指令和数据。ASIDE将固定的垂直旋转应用于数据令牌的嵌入，从而在不引入任何额外参数的情况下创建指令和数据令牌的不同表示。我们通过使用ASIDE对LLM进行预算调整来证明我们方法的有效性，并显示（1）在不损失模型能力的情况下大幅提高的描述数据分离分数，以及（2）即使没有专门的安全培训，也在即时注入基准上具有竞争力的结果。此外，我们通过分析模型表示来研究我们方法背后的工作机制。



## **9. MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning**

Mr. Guard：使用课程学习的多语言推理保护 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15241v1) [paper-pdf](http://arxiv.org/pdf/2504.15241v1)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.

摘要: 大型语言模型（LLM）容易受到诸如越狱之类的对抗性攻击，这可能会引发有害或不安全的行为。在多语言环境中，这种漏洞会加剧，因为多语言安全一致的数据通常有限。因此，开发一个能够检测和过滤不同语言的不安全内容的护栏对于在现实世界的应用程序中部署LLM至关重要。在这项工作中，我们提出了一种方法来建立一个多语言护栏推理。我们的方法包括：（1）综合多语言数据生成，融合了文化和语言上的细微差别，（2）监督微调，以及（3）课程引导的群组相对政策优化（GRPO）框架，进一步提高性能。实验结果表明，我们的多语言护栏在域内和域外语言中始终优于最近的基线。我们护栏的多语言推理能力使其能够生成多语言解释，这对于理解多语言内容审核中的特定语言风险和歧义特别有用。



## **10. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

HiddenDetect：通过监视隐藏状态检测针对大型视觉语言模型的越狱攻击 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.14744v3) [paper-pdf](http://arxiv.org/pdf/2502.14744v3)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.

摘要: 与纯语言模型相比，其他模式的集成增加了大型视觉语言模型（LVLM）对安全风险（如越狱攻击）的敏感性。虽然现有的研究主要集中在事后对齐技术，LVLM内的潜在安全机制仍然在很大程度上未被探索。在这项工作中，我们调查是否LVLM内在编码安全相关的信号在其内部激活过程中的推理。我们的研究结果表明，LVLM在处理不安全的提示时表现出不同的激活模式，可以利用它来检测和减轻对抗性输入，而不需要进行广泛的微调。基于这一见解，我们引入了HiddenDetect，这是一个新颖的免调框架，可以利用内部模型激活来增强安全性。实验结果表明，{HiddenDetect}在检测针对LVLM的越狱攻击方面超越了最先进的方法。通过利用固有的安全感知模式，我们的方法提供了一种高效且可扩展的解决方案，用于加强LVLM针对多模式威胁的鲁棒性。我们的代码将在https://github.com/leigest519/HiddenDetect上公开发布。



## **11. RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search**

RainbowPlus：通过进化质量多样性搜索增强对抗提示生成 cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15047v1) [paper-pdf](http://arxiv.org/pdf/2504.15047v1)

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at https://github.com/knoveleng/rainbowplus, supporting reproducibility and future research in LLM red-teaming.

摘要: 大型语言模型（LLM）表现出非凡的能力，但很容易受到对抗提示的影响，这些提示利用漏洞产生不安全或有偏见的输出。现有的红色团队方法通常面临可扩展性挑战、资源密集型要求或攻击策略的多样性有限。我们提出RainbowPlus，这是一种植根于进化计算的新型红色团队框架，通过自适应质量多样性（QD）搜索来增强对抗提示生成，该搜索扩展了MAP-Elites等经典进化算法，并为语言模型量身定制的创新。通过采用多元素档案来存储各种高质量提示，并采用全面的适应度函数来同时评估多个提示，RainbowPlus克服了Rainbow Teaming等现有QD方法中单提示档案和成对比较的限制。在六个基准数据集和四个开源LLM上比较RainbowPlus与QD方法的实验证明了卓越的攻击成功率（ASB）和多样性（Diverse-Score $\大约0.84$），生成高达100倍的独特提示（例如，Ministral-8B-Direct-2410为10，418 vs 100）。与HarmBench数据集中的九种最先进方法（12个LLM（10个开源，2个封闭源）相比，RainbowPlus的平均ASB为81.1%，超过AutoDAN-Turbo 3.9%，速度快9倍（1.45小时vs 13.50小时）。我们的开源实施促进了LLM安全性的进一步进步，为漏洞评估提供了可扩展的工具。代码和资源可在https：//github.com/knoveleng/rainbowplus上公开获取，支持LLM红色团队的再现性和未来研究。



## **12. aiXamine: LLM Safety and Security Simplified**

aiXamine：LLM安全与安保简化 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14985v1) [paper-pdf](http://arxiv.org/pdf/2504.14985v1)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.

摘要: 评估大型语言模型（LLM）的安全性和保障性仍然是一项复杂的任务，通常需要用户在临时基准、数据集、指标和报告格式的碎片化环境中进行导航。为了应对这一挑战，我们推出了aiXamine，这是一个针对LLM安全性的全面黑匣子评估平台。aiXamine集成了40多个测试（即，基准）组织成八个关键服务，针对安全和保障的特定维度：对抗稳健性、代码安全性、公平性和偏见、幻觉、模型和数据隐私、分发外（OOD）稳健性、过度拒绝和安全对齐。该平台将评估结果汇总到每个模型的单个详细报告中，提供模型性能、测试示例和丰富的可视化的详细细分。我们使用aiXamine评估了50多个公开和专有的LLM，进行了超过2000次检查。我们的研究结果揭示了领先模型中的显着漏洞，包括OpenAI GPT-4 o中容易受到对抗攻击、xAI Grok-3中的偏见输出以及Google Gemini 2.0中的隐私弱点。此外，我们观察到开源模型可以在特定服务中匹配或超过专有模型，例如安全性一致、公平性和偏差以及OOD稳健性。最后，我们确定了蒸馏策略、模型大小、训练方法和架构选择之间的权衡。



## **13. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

在智能电网中实践大型语言模型的风险：威胁建模和验证 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2405.06237v3) [paper-pdf](http://arxiv.org/pdf/2405.06237v3)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large language models (LLMs) represent significant breakthroughs in artificial intelligence and hold potential for applications within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluated the risks of LLMs and identified two major types of attacks relevant to potential smart grid LLM applications, presenting the corresponding threat models. We validated these attacks using popular LLMs and real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in different smart grid applications.

摘要: 大型语言模型（LLM）代表了人工智能的重大突破，具有智能电网中的应用潜力。然而，正如之前的文献所证明的那样，人工智能技术容易受到各种类型的攻击。在将LLM部署在智能电网等关键基础设施中之前，调查和评估与LLM相关的风险至关重要。在本文中，我们系统地评估了LLM的风险，并识别了与潜在智能电网LLM应用相关的两种主要攻击类型，并给出了相应的威胁模型。我们使用流行的LLM和真实的智能电网数据验证了这些攻击。我们的验证表明，攻击者能够从不同智能电网应用程序中使用的LLM中注入不良数据并检索领域知识。



## **14. MCGMark: An Encodable and Robust Online Watermark for Tracing LLM-Generated Malicious Code**

MCGMark：一种用于跟踪LLM恶意代码的可编码鲁棒在线水印 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2408.01354v2) [paper-pdf](http://arxiv.org/pdf/2408.01354v2)

**Authors**: Kaiwen Ning, Jiachi Chen, Qingyuan Zhong, Tao Zhang, Yanlin Wang, Wei Li, Jingwen Zhang, Jianxing Yu, Yuming Feng, Weizhe Zhang, Zibin Zheng

**Abstract**: With the advent of large language models (LLMs), numerous software service providers (SSPs) are dedicated to developing LLMs customized for code generation tasks, such as CodeLlama and Copilot. However, these LLMs can be leveraged by attackers to create malicious software, which may pose potential threats to the software ecosystem. For example, they can automate the creation of advanced phishing malware. To address this issue, we first conduct an empirical study and design a prompt dataset, MCGTest, which involves approximately 400 person-hours of work and consists of 406 malicious code generation tasks. Utilizing this dataset, we propose MCGMark, the first robust, code structure-aware, and encodable watermarking approach to trace LLM-generated code. We embed encodable information by controlling the token selection and ensuring the output quality based on probabilistic outliers. Additionally, we enhance the robustness of the watermark by considering the structural features of malicious code, preventing the embedding of the watermark in easily modified positions, such as comments. We validate the effectiveness and robustness of MCGMark on the DeepSeek-Coder. MCGMark achieves an embedding success rate of 88.9% within a maximum output limit of 400 tokens. Furthermore, it also demonstrates strong robustness and has minimal impact on the quality of the output code. Our approach assists SSPs in tracing and holding responsible parties accountable for malicious code generated by LLMs.

摘要: 随着大型语言模型（LLM）的出现，许多软件服务提供商（SSP）致力于开发为代码生成任务定制的LLM，例如CodeLlama和Copilot。然而，攻击者可以利用这些LLM创建恶意软件，这可能对软件生态系统构成潜在威胁。例如，他们可以自动创建高级网络钓鱼恶意软件。为了解决这个问题，我们首先进行了一项实证研究并设计了一个即时数据集MCGTest，该数据集涉及大约400个工时的工作，由406个恶意代码生成任务组成。利用该数据集，我们提出了MCGMark，这是第一个鲁棒的、代码结构感知的、可编码的水印方法，用于跟踪LLM生成的代码。我们通过控制令牌选择并基于概率异常值确保输出质量来嵌入可编码信息。此外，我们通过考虑恶意代码的结构特征来增强水印的鲁棒性，防止将水印嵌入在易于修改的位置（例如评论）。我们在DeepSeek-Coder上验证了MCGMark的有效性和稳健性。MCGMark在400个代币的最大输出限制内实现了88.9%的嵌入成功率。此外，它还表现出很强的鲁棒性，并且对输出代码质量的影响最小。我们的方法可帮助ISP追踪LLM生成的恶意代码并追究责任方的责任。



## **15. BadApex: Backdoor Attack Based on Adaptive Optimization Mechanism of Black-box Large Language Models**

BadApex：基于黑匣子大型语言模型自适应优化机制的后门攻击 cs.CL

16 pages, 6 figures

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.13775v2) [paper-pdf](http://arxiv.org/pdf/2504.13775v2)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Ziwei Zhang, Yinghan Zhou, Yiming Xue

**Abstract**: Previous insertion-based and paraphrase-based backdoors have achieved great success in attack efficacy, but they ignore the text quality and semantic consistency between poisoned and clean texts. Although recent studies introduce LLMs to generate poisoned texts and improve the stealthiness, semantic consistency, and text quality, their hand-crafted prompts rely on expert experiences, facing significant challenges in prompt adaptability and attack performance after defenses. In this paper, we propose a novel backdoor attack based on adaptive optimization mechanism of black-box large language models (BadApex), which leverages a black-box LLM to generate poisoned text through a refined prompt. Specifically, an Adaptive Optimization Mechanism is designed to refine an initial prompt iteratively using the generation and modification agents. The generation agent generates the poisoned text based on the initial prompt. Then the modification agent evaluates the quality of the poisoned text and refines a new prompt. After several iterations of the above process, the refined prompt is used to generate poisoned texts through LLMs. We conduct extensive experiments on three dataset with six backdoor attacks and two defenses. Extensive experimental results demonstrate that BadApex significantly outperforms state-of-the-art attacks. It improves prompt adaptability, semantic consistency, and text quality. Furthermore, when two defense methods are applied, the average attack success rate (ASR) still up to 96.75%.

摘要: 之前的基于插入和基于转述的后门在攻击功效方面取得了巨大成功，但它们忽视了有毒文本和干净文本之间的文本质量和语义一致性。尽管最近的研究引入了LLM来生成有毒文本并提高隐蔽性、语义一致性和文本质量，但其手工制作的提示依赖于专家经验，在防御后的即时适应性和攻击性能方面面临着重大挑战。本文提出了一种基于黑匣子大型语言模型（BadApex）的自适应优化机制的新型后门攻击，该攻击利用黑匣子LLM通过细化的提示生成有毒文本。具体来说，自适应优化机制旨在使用生成和修改代理迭代地细化初始提示。生成代理根据初始提示生成中毒文本。然后修改代理评估中毒文本的质量并精炼新的提示。经过上述过程的多次迭代后，使用改进的提示通过LLM生成有毒文本。我们对三个数据集进行了广泛的实验，其中包含六种后门攻击和两种防御。大量实验结果表明，BadApex的性能明显优于最先进的攻击。它提高了即时适应性、语义一致性和文本质量。此外，当采用两种防御方法时，平均攻击成功率（ASB）仍高达96.75%。



## **16. Detecting Training Data of Large Language Models via Expectation Maximization**

基于期望最大化的大型语言模型训练数据检测 cs.CL

15 pages

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.07582v2) [paper-pdf](http://arxiv.org/pdf/2410.07582v2)

**Authors**: Gyuwan Kim, Yang Li, Evangelia Spiliopoulou, Jie Ma, Miguel Ballesteros, William Yang Wang

**Abstract**: The advancement of large language models has grown parallel to the opacity of their training data. Membership inference attacks (MIAs) aim to determine whether specific data was used to train a model. They offer valuable insights into detecting data contamination and ensuring compliance with privacy and copyright standards. However, MIA for LLMs is challenging due to the massive scale of training data and the inherent ambiguity of membership in texts. Moreover, creating realistic MIA evaluation benchmarks is difficult as training and test data distributions are often unknown. We introduce EM-MIA, a novel membership inference method that iteratively refines membership scores and prefix scores via an expectation-maximization algorithm. Our approach leverages the observation that these scores can improve each other: membership scores help identify effective prefixes for detecting training data, while prefix scores help determine membership. As a result, EM-MIA achieves state-of-the-art results on WikiMIA. To enable comprehensive evaluation, we introduce OLMoMIA, a benchmark built from OLMo resources, which allows controlling task difficulty through varying degrees of overlap between training and test data distributions. Our experiments demonstrate EM-MIA is robust across different scenarios while also revealing fundamental limitations of current MIA approaches when member and non-member distributions are nearly identical.

摘要: 大型语言模型的进步与其训练数据的不透明同步发展。隶属度推理攻击（MIA）旨在确定是否使用特定数据来训练模型。它们为检测数据污染并确保遵守隐私和版权标准提供了宝贵的见解。然而，由于训练数据规模庞大以及文本成员资格固有的模糊性，LLM的MIA具有挑战性。此外，创建现实的MIA评估基准很困难，因为训练和测试数据分布通常是未知的。我们引入了EM-MIA，这是一种新型的成员资格推理方法，通过期望最大化算法迭代地细化成员资格分数和前置分数。我们的方法利用了这些分数可以相互改进的观察：成员资格分数有助于识别用于检测训练数据的有效前置码，而前置码分数有助于确定成员资格。因此，EM-MIA在WikiMIA上实现了最先进的结果。为了实现全面评估，我们引入了OLMoMIA，这是一个从OLMo资源构建的基准，它允许通过训练和测试数据分布之间不同程度的重叠来控制任务难度。我们的实验证明，EM-MIA在不同场景中具有鲁棒性，同时也揭示了当前MIA方法在成员和非成员分布几乎相同时的根本局限性。



## **17. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents**

提示流程完整性以防止LLM代理的特权升级 cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.15547v2) [paper-pdf](http://arxiv.org/pdf/2503.15547v2)

**Authors**: Juhee Kim, Woohyuk Choi, Byoungyoung Lee

**Abstract**: Large Language Models (LLMs) are combined with tools to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or tool's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompts are prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., agent isolation, secure untrusted data processing, and privilege escalation guardrails. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents.

摘要: 大型语言模型（LLM）与工具相结合，创建强大的LLM代理，提供广泛的服务。与传统软件不同，LLM代理的行为在运行时由来自用户或工具数据的自然语言提示确定。这种灵活性可以实现具有无限功能和可编程性的新计算范式，但也引入了新的安全风险，容易受到特权升级攻击。此外，用户提示很容易被LLM代理以不安全的方式解释，从而创建可被攻击者利用的非确定性行为。为了解决这些安全风险，我们提出了提示流完整性（PFI），这是一种面向系统安全的解决方案，用于防止LLM代理中的特权升级。PFI分析了LLM代理的体系结构特征，具有三种缓解技术--即，代理隔离、安全的不可信数据处理和特权升级护栏。我们的评估结果表明，PFI有效地缓解了特权升级攻击，同时成功保留了LLM代理的实用性。



## **18. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

大型语言模型作为软件分析中稳健的数据生成器：我们已经做到了吗？ cs.SE

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2411.10565v2) [paper-pdf](http://arxiv.org/pdf/2411.10565v2)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.

摘要: 大型语言模型（LLM）生成的数据越来越多地用于软件分析，但目前尚不清楚该数据与人类编写的数据相比如何，特别是当模型暴露于对抗场景时。对抗性攻击可能会损害软件系统的可靠性和安全性，因此，与作为模型性能基准的人类编写数据相比，了解LLM生成的数据在这些条件下的表现如何，可以为LLM生成的数据是否提供类似的稳健性和有效性提供有价值的见解。为了解决这一差距，我们系统地评估和比较人类编写的数据和LLM生成的数据的质量，以便在对抗性攻击的背景下微调稳健的预训练模型（Ptms）。我们评估了六种广泛使用的PtM的稳健性，这些PtM在对抗性攻击之前和之后根据人类编写和LLM生成的数据进行了微调。该评估在三个流行的软件分析任务中使用了九种最先进的（SOTA）对抗性攻击技术：克隆检测，代码摘要和代码审查讨论中的情感分析。此外，我们使用11个相似性度量来分析生成的对抗性示例的质量。我们的研究结果表明，虽然对LLM生成的数据进行微调的PTM与对人类编写的数据进行微调的PTM具有竞争力，但它们在软件分析任务中对对抗性攻击的鲁棒性较低。我们的研究强调了进一步探索提高LLM生成的训练数据质量的必要性，以开发高性能且能够抵御软件分析中的对抗性攻击的模型。



## **19. REDEditing: Relationship-Driven Precise Backdoor Poisoning on Text-to-Image Diffusion Models**

Redediting：广告驱动的文本到图像扩散模型上的精确后门中毒 cs.CR

10 pages, 7 figures

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2504.14554v1) [paper-pdf](http://arxiv.org/pdf/2504.14554v1)

**Authors**: Chongye Guo, Jinhu Fu, Junfeng Fang, Kun Wang, Guorui Feng

**Abstract**: The rapid advancement of generative AI highlights the importance of text-to-image (T2I) security, particularly with the threat of backdoor poisoning. Timely disclosure and mitigation of security vulnerabilities in T2I models are crucial for ensuring the safe deployment of generative models. We explore a novel training-free backdoor poisoning paradigm through model editing, which is recently employed for knowledge updating in large language models. Nevertheless, we reveal the potential security risks posed by model editing techniques to image generation models. In this work, we establish the principles for backdoor attacks based on model editing, and propose a relationship-driven precise backdoor poisoning method, REDEditing. Drawing on the principles of equivalent-attribute alignment and stealthy poisoning, we develop an equivalent relationship retrieval and joint-attribute transfer approach that ensures consistent backdoor image generation through concept rebinding. A knowledge isolation constraint is proposed to preserve benign generation integrity. Our method achieves an 11\% higher attack success rate compared to state-of-the-art approaches. Remarkably, adding just one line of code enhances output naturalness while improving backdoor stealthiness by 24\%. This work aims to heighten awareness regarding this security vulnerability in editable image generation models.

摘要: 生成性人工智能的快速发展凸显了文本到图像（T2 I）安全的重要性，特别是在后门中毒的威胁下。及时披露和缓解T2 I模型中的安全漏洞对于确保生成模型的安全部署至关重要。我们通过模型编辑探索了一种新型的免训练后门中毒范式，该范式最近被用于大型语言模型中的知识更新。尽管如此，我们揭示了模型编辑技术对图像生成模型构成的潜在安全风险。在这项工作中，我们建立了基于模型编辑的后门攻击的原则，并提出了一种关系驱动的精确后门中毒方法--REDEditing。利用等效属性对齐和隐形中毒的原则，我们开发了一种等效关系检索和联合属性转移方法，通过概念重新绑定确保一致的后门图像生成。提出了知识隔离约束以保持良性生成完整性。与最先进的方法相比，我们的方法的攻击成功率高出11%。值得注意的是，仅添加一行代码就可以增强输出的自然性，同时将后门的隐蔽性提高了24%。这项工作旨在提高人们对可编辑图像生成模型中此安全漏洞的认识。



## **20. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

通过大语言模型重建差异私人文本清理 cs.CR

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2410.12443v2) [paper-pdf](http://arxiv.org/pdf/2410.12443v2)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.

摘要: 差异隐私（DP）是针对隐私泄露攻击的事实上的隐私标准，包括最近发现的许多针对大型语言模型（LLM）的攻击。然而，我们发现LLM可以从给定的DP消毒提示中重建更改/删除的隐私。我们基于LLM的可访问性提出了两种攻击（黑匣子和白盒），并表明LLM可以通过提供样本文本对作为指令（在黑匣子攻击中）或微调数据（在白盒攻击中）来连接DP清理文本对和LLM的相应私人训练数据。为了说明我们的发现，我们对现代LLM进行了全面的实验（例如，LLaMA-2、LLaMA-3、ChatGPT-3.5、ChatGPT-4、ChatGPT-4 o、Claude-3、Claude-3.5、OPT、GPT-Neo、GPT-J、Gemma-2和Pythia）针对单词级和业务级DP使用常用数据集（例如WikiMIA、Pile-CC和Pile-iki）。实验结果显示出有希望的回收率，例如，针对WikiMIA数据集的词级DP的黑匣子攻击在LLaMA-2（70 B）上为72.18%，在LLaMA-3（70 B）上为82.39%，在Gemma-2上为75.35%，在ChatGPT-4 o上为91.2%，在Claude-3.5（十四行诗）上为94.01%。更紧迫的是，这项研究表明，这些著名的LLM已成为当前环境下现有DP文本清理方法的新安全风险。



## **21. SHIELD : An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models**

SHIELD：使用多模式大型语言模型进行面部欺骗和伪造检测的评估基准 cs.CV

Accepted by Visual Intelligence

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2402.04178v2) [paper-pdf](http://arxiv.org/pdf/2402.04178v2)

**Authors**: Yichen Shi, Yuhao Gao, Yingxin Lai, Hongyang Wang, Jun Feng, Lei He, Jun Wan, Changsheng Chen, Zitong Yu, Xiaochun Cao

**Abstract**: Multimodal large language models (MLLMs) have demonstrated strong capabilities in vision-related tasks, capitalizing on their visual semantic comprehension and reasoning capabilities. However, their ability to detect subtle visual spoofing and forgery clues in face attack detection tasks remains underexplored. In this paper, we introduce a benchmark, SHIELD, to evaluate MLLMs for face spoofing and forgery detection. Specifically, we design true/false and multiple-choice questions to assess MLLM performance on multimodal face data across two tasks. For the face anti-spoofing task, we evaluate three modalities (i.e., RGB, infrared, and depth) under six attack types. For the face forgery detection task, we evaluate GAN-based and diffusion-based data, incorporating visual and acoustic modalities. We conduct zero-shot and few-shot evaluations in standard and chain of thought (COT) settings. Additionally, we propose a novel multi-attribute chain of thought (MA-COT) paradigm for describing and judging various task-specific and task-irrelevant attributes of face images. The findings of this study demonstrate that MLLMs exhibit strong potential for addressing the challenges associated with the security of facial recognition technology applications.

摘要: 多模态大型语言模型（MLLM）在视觉相关任务中表现出强大的能力，利用其视觉语义理解和推理能力。然而，它们在人脸攻击检测任务中检测微妙的视觉欺骗和伪造线索的能力仍然有待探索。在本文中，我们介绍了一个基准，SHIELD，评估MLLM的人脸欺骗和伪造检测。具体来说，我们设计了真/假和多项选择题来评估MLLM在两个任务中对多模态人脸数据的性能。对于面部反欺骗任务，我们评估了三种模式（即，GB、红外和深度）六种攻击类型。对于面部伪造检测任务，我们评估基于GAN和基于扩散的数据，并结合视觉和声学模式。我们在标准和思想链（COT）环境中进行零射击和少射击评估。此外，我们提出了一种新型的多属性思维链（MA-COT）范式，用于描述和判断面部图像的各种特定任务和任务无关的属性。这项研究的结果表明，MLLM在解决与面部识别技术应用安全相关的挑战方面具有强大的潜力。



## **22. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

Accepted to ICLR 2025. Code:  https://github.com/zhxieml/remiss-jailbreak

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2406.14393v5) [paper-pdf](http://arxiv.org/pdf/2406.14393v5)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.

摘要: 大型语言模型（LLM）的广泛采用引发了人们对其安全性和可靠性的担忧，特别是对其容易受到对抗攻击的影响。在本文中，我们提出了一种新颖的视角，将此漏洞归因于对齐过程中的奖励错误指定。当奖励函数未能准确捕捉预期行为时，就会发生这种错误规范，从而导致模型输出不一致。我们引入了一个指标ReGap来量化奖励错误指定的程度，并展示其在检测有害后门提示方面的有效性和稳健性。在这些见解的基础上，我们介绍了ReMiss，这是一个自动化红色团队系统，可以在奖励错误指定的空间中生成对抗提示。ReMiss在AdvBench基准上针对各种目标对齐的LLM实现了最先进的攻击成功率，同时保留了生成提示的人类可读性。此外，这些对开源模型的攻击证明了对GPT-4 o等闭源模型和HarmBench的分发外任务的高度可移植性。详细的分析强调了与之前的方法相比，拟议的奖励错误指定目标的独特优势，为提高LLM的安全性和稳健性提供了新的见解。



## **23. Multi-Stage Retrieval for Operational Technology Cybersecurity Compliance Using Large Language Models: A Railway Casestudy**

使用大型语言模型的运营技术网络安全合规性多阶段检索：铁路案例研究 cs.AI

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.14044v1) [paper-pdf](http://arxiv.org/pdf/2504.14044v1)

**Authors**: Regan Bolton, Mohammadreza Sheikhfathollahi, Simon Parkinson, Dan Basher, Howard Parkinson

**Abstract**: Operational Technology Cybersecurity (OTCS) continues to be a dominant challenge for critical infrastructure such as railways. As these systems become increasingly vulnerable to malicious attacks due to digitalization, effective documentation and compliance processes are essential to protect these safety-critical systems. This paper proposes a novel system that leverages Large Language Models (LLMs) and multi-stage retrieval to enhance the compliance verification process against standards like IEC 62443 and the rail-specific IEC 63452. We first evaluate a Baseline Compliance Architecture (BCA) for answering OTCS compliance queries, then develop an extended approach called Parallel Compliance Architecture (PCA) that incorporates additional context from regulatory standards. Through empirical evaluation comparing OpenAI-gpt-4o and Claude-3.5-haiku models in these architectures, we demonstrate that the PCA significantly improves both correctness and reasoning quality in compliance verification. Our research establishes metrics for response correctness, logical reasoning, and hallucination detection, highlighting the strengths and limitations of using LLMs for compliance verification in railway cybersecurity. The results suggest that retrieval-augmented approaches can significantly improve the efficiency and accuracy of compliance assessments, particularly valuable in an industry facing a shortage of cybersecurity expertise.

摘要: 运营技术网络安全（OTCS）仍然是铁路等关键基础设施的主要挑战。随着这些系统因数字化而变得越来越容易受到恶意攻击，有效的文档和合规流程对于保护这些安全关键系统至关重要。本文提出了一种新颖的系统，该系统利用大型语言模型（LLM）和多阶段检索来增强针对IEC 62443和铁路特定IEC 63452等标准的合规性验证过程。我们首先评估用于回答OTCS合规性查询的基线合规架构（BCA），然后开发一种名为并行合规架构（PCA）的扩展方法，该方法结合了监管标准的额外上下文。通过比较这些架构中的OpenAI-gpt-4 o和Claude-3.5-俳句模型的实证评估，我们证明PCA显着提高了合规性验证的正确性和推理质量。我们的研究建立了响应正确性、逻辑推理和幻觉检测的指标，强调了使用LLM进行铁路网络安全合规性验证的优点和局限性。结果表明，检索增强方法可以显着提高合规性评估的效率和准确性，在面临网络安全专业知识短缺的行业中尤其有价值。



## **24. Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?**

使用LLM检测PyPI包中的恶意源代码：RAG方便吗？ cs.SE

The paper has been peer-reviewed and accepted for publication to the  29th International Conference on Evaluation and Assessment in Software  Engineering (EASE 2025)

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13769v1) [paper-pdf](http://arxiv.org/pdf/2504.13769v1)

**Authors**: Motunrayo Ibiyo, Thinakone Louangdy, Phuong T. Nguyen, Claudio Di Sipio, Davide Di Ruscio

**Abstract**: Malicious software packages in open-source ecosystems, such as PyPI, pose growing security risks. Unlike traditional vulnerabilities, these packages are intentionally designed to deceive users, making detection challenging due to evolving attack methods and the lack of structured datasets. In this work, we empirically evaluate the effectiveness of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and few-shot learning for detecting malicious source code. We fine-tune LLMs on curated datasets and integrate YARA rules, GitHub Security Advisories, and malicious code snippets with the aim of enhancing classification accuracy. We came across a counterintuitive outcome: While RAG is expected to boost up the prediction performance, it fails in the performed evaluation, obtaining a mediocre accuracy. In contrast, few-shot learning is more effective as it significantly improves the detection of malicious code, achieving 97% accuracy and 95% balanced accuracy, outperforming traditional RAG approaches. Thus, future work should expand structured knowledge bases, refine retrieval models, and explore hybrid AI-driven cybersecurity solutions.

摘要: 开源生态系统中的恶意软件包（例如PyPI）带来了越来越大的安全风险。与传统漏洞不同，这些包是故意设计来欺骗用户的，由于攻击方法不断发展和缺乏结构化数据集，检测变得具有挑战性。在这项工作中，我们根据经验评估了大型语言模型（LLM）、检索增强生成（RAG）和少量学习检测恶意源代码的有效性。我们对精心策划的数据集进行微调，并集成YARA规则、GitHub安全建议和恶意代码片段，旨在提高分类准确性。我们遇到了一个违反直觉的结果：虽然RAG有望提高预测性能，但它在执行的评估中失败了，获得了平庸的准确性。相比之下，少量学习更有效，因为它显著提高了恶意代码的检测，达到了97%的准确率和95%的平衡准确率，优于传统的RAG方法。因此，未来的工作应该扩展结构化知识库，完善检索模型，并探索混合人工智能驱动的网络安全解决方案。



## **25. DETAM: Defending LLMs Against Jailbreak Attacks via Targeted Attention Modification**

SEARCH：通过定向注意力修改保护LLM免受越狱攻击 cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13562v1) [paper-pdf](http://arxiv.org/pdf/2504.13562v1)

**Authors**: Yu Li, Han Jiang, Zhihua Wei

**Abstract**: With the widespread adoption of Large Language Models (LLMs), jailbreak attacks have become an increasingly pressing safety concern. While safety-aligned LLMs can effectively defend against normal harmful queries, they remain vulnerable to such attacks. Existing defense methods primarily rely on fine-tuning or input modification, which often suffer from limited generalization and reduced utility. To address this, we introduce DETAM, a finetuning-free defense approach that improves the defensive capabilities against jailbreak attacks of LLMs via targeted attention modification. Specifically, we analyze the differences in attention scores between successful and unsuccessful defenses to identify the attention heads sensitive to jailbreak attacks. During inference, we reallocate attention to emphasize the user's core intention, minimizing interference from attack tokens. Our experimental results demonstrate that DETAM outperforms various baselines in jailbreak defense and exhibits robust generalization across different attacks and models, maintaining its effectiveness even on in-the-wild jailbreak data. Furthermore, in evaluating the model's utility, we incorporated over-defense datasets, which further validate the superior performance of our approach. The code will be released immediately upon acceptance.

摘要: 随着大型语言模型（LLM）的广泛采用，越狱攻击已成为一个日益紧迫的安全问题。虽然安全一致的LLM可以有效地防御正常的有害查询，但它们仍然容易受到此类攻击。现有的防御方法主要依赖于微调或输入修改，这通常会受到通用性有限和实用性降低的影响。为了解决这个问题，我们引入了SEARCH，这是一种无微调的防御方法，通过有针对性的注意力修改来提高针对LLM越狱攻击的防御能力。具体来说，我们分析成功和不成功防御之间注意力分数的差异，以识别对越狱攻击敏感的注意力头。在推理过程中，我们重新分配注意力以强调用户的核心意图，最大限度地减少攻击令牌的干扰。我们的实验结果表明，DeliverM在越狱防御方面的表现优于各种基线，并且在不同的攻击和模型之间表现出强大的概括性，即使在野外越狱数据上也保持其有效性。此外，在评估模型的效用时，我们纳入了过度防御数据集，这进一步验证了我们方法的卓越性能。该代码将在接受后立即发布。



## **26. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

WWW'25 research track accepted

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2406.11260v3) [paper-pdf](http://arxiv.org/pdf/2406.11260v3)

**Authors**: Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, Meeyoung Cha

**Abstract**: The spread of fake news harms individuals and presents a critical social challenge that must be addressed. Although numerous algorithmic and insightful features have been developed to detect fake news, many of these features can be manipulated with style-conversion attacks, especially with the emergence of advanced language models, making it more difficult to differentiate from genuine news. This study proposes adversarial style augmentation, AdStyle, designed to train a fake news detector that remains robust against various style-conversion attacks. The primary mechanism involves the strategic use of LLMs to automatically generate a diverse and coherent array of style-conversion attack prompts, enhancing the generation of particularly challenging prompts for the detector. Experiments indicate that our augmentation strategy significantly improves robustness and detection performance when evaluated on fake news benchmark datasets.

摘要: 假新闻的传播伤害了个人，并提出了必须解决的严重社会挑战。尽管已经开发了许多算法和有洞察力的功能来检测假新闻，但其中许多功能都可以通过风格转换攻击来操纵，特别是随着高级语言模型的出现，使其更难与真实新闻区分开来。这项研究提出了对抗性风格增强AdStyle，旨在训练假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。主要机制涉及战略性地使用LLM来自动生成多样化且连贯的风格转换攻击提示阵列，从而增强检测器特别具有挑战性的提示的生成。实验表明，当对假新闻基准数据集进行评估时，我们的增强策略显着提高了鲁棒性和检测性能。



## **27. Large Language Models for Validating Network Protocol Parsers**

用于验证网络协议解析器的大型语言模型 cs.SE

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13515v1) [paper-pdf](http://arxiv.org/pdf/2504.13515v1)

**Authors**: Mingwei Zheng, Danning Xie, Xiangyu Zhang

**Abstract**: Network protocol parsers are essential for enabling correct and secure communication between devices. Bugs in these parsers can introduce critical vulnerabilities, including memory corruption, information leakage, and denial-of-service attacks. An intuitive way to assess parser correctness is to compare the implementation with its official protocol standard. However, this comparison is challenging because protocol standards are typically written in natural language, whereas implementations are in source code. Existing methods like model checking, fuzzing, and differential testing have been used to find parsing bugs, but they either require significant manual effort or ignore the protocol standards, limiting their ability to detect semantic violations. To enable more automated validation of parser implementations against protocol standards, we propose PARVAL, a multi-agent framework built on large language models (LLMs). PARVAL leverages the capabilities of LLMs to understand both natural language and code. It transforms both protocol standards and their implementations into a unified intermediate representation, referred to as format specifications, and performs a differential comparison to uncover inconsistencies. We evaluate PARVAL on the Bidirectional Forwarding Detection (BFD) protocol. Our experiments demonstrate that PARVAL successfully identifies inconsistencies between the implementation and its RFC standard, achieving a low false positive rate of 5.6%. PARVAL uncovers seven unique bugs, including five previously unknown issues.

摘要: 网络协议解析器对于实现设备之间正确、安全的通信至关重要。这些解析器中的错误可能会引入关键漏洞，包括内存损坏、信息泄露和拒绝服务攻击。评估解析器正确性的直观方法是将实现与其官方协议标准进行比较。然而，这种比较具有挑战性，因为协议标准通常是用自然语言编写的，而实现是用源代码编写的。模型检查、模糊化和差异测试等现有方法已被用来发现解析错误，但它们要么需要大量的手动工作，要么忽略协议标准，从而限制了它们检测语义违规的能力。为了能够根据协议标准对解析器实现进行更自动化的验证，我们提出了PARVAR，这是一个基于大型语言模型（LLM）的多代理框架。PARVAR利用LLM的功能来理解自然语言和代码。它将协议标准及其实现转换为统一的中间表示（称为格式规范），并执行差异比较以发现不一致之处。我们在双向转发检测（Illustrator）协议上评估PARVAR。我们的实验表明，PARVAR成功地识别了实现与其RFC标准之间的不一致之处，实现了5.6%的低假阳性率。PARVAR发现了七个独特的错误，其中包括五个以前未知的问题。



## **28. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.05050v2) [paper-pdf](http://arxiv.org/pdf/2504.05050v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **29. GraphQLer: Enhancing GraphQL Security with Context-Aware API Testing**

GraphQLer：通过上下文感知API测试增强GraphQL安全性 cs.CR

Publicly available on: https://github.com/omar2535/GraphQLer

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13358v1) [paper-pdf](http://arxiv.org/pdf/2504.13358v1)

**Authors**: Omar Tsai, Jianing Li, Tsz Tung Cheung, Lejing Huang, Hao Zhu, Jianrui Xiao, Iman Sharafaldin, Mohammad A. Tayebi

**Abstract**: GraphQL is an open-source data query and manipulation language for web applications, offering a flexible alternative to RESTful APIs. However, its dynamic execution model and lack of built-in security mechanisms expose it to vulnerabilities such as unauthorized data access, denial-of-service (DoS) attacks, and injections. Existing testing tools focus on functional correctness, often overlooking security risks stemming from query interdependencies and execution context. This paper presents GraphQLer, the first context-aware security testing framework for GraphQL APIs. GraphQLer constructs a dependency graph to analyze relationships among mutations, queries, and objects, capturing critical interdependencies. It chains related queries and mutations to reveal authentication and authorization flaws, access control bypasses, and resource misuse. Additionally, GraphQLer tracks internal resource usage to uncover data leakage, privilege escalation, and replay attack vectors. We assess GraphQLer on various GraphQL APIs, demonstrating improved testing coverage - averaging a 35% increase, with up to 84% in some cases - compared to top-performing baselines. Remarkably, this is achieved in less time, making GraphQLer suitable for time-sensitive contexts. GraphQLer also successfully detects a known CVE and potential vulnerabilities in large-scale production APIs. These results underline GraphQLer's utility in proactively securing GraphQL APIs through automated, context-aware vulnerability detection.

摘要: GraphQL是一种用于Web应用程序的开源数据查询和操作语言，提供了RESTful API的灵活替代方案。然而，它的动态执行模型和缺乏内置安全机制使其面临未经授权的数据访问、拒绝服务（DPS）攻击和注入等漏洞。现有的测试工具专注于功能正确性，通常忽视了由查询相互依赖性和执行上下文产生的安全风险。本文介绍了GraphQLer，这是第一个针对GraphQL API的上下文感知安全测试框架。GraphQLer构建了一个依赖图来分析变化、查询和对象之间的关系，捕获关键的相互依赖关系。它链接相关的查询和变化，以揭示身份验证和授权缺陷，访问控制绕过和资源滥用。此外，GraphQLer还跟踪内部资源使用情况，以发现数据泄漏、权限提升和重放攻击向量。我们在各种GraphQL API上对GraphQLer进行了评估，与性能最好的基线相比，测试覆盖率平均提高了35%，在某些情况下高达84%。值得注意的是，这可以在更短的时间内实现，使GraphQLer适合时间敏感的上下文。GraphQLer还成功检测到大规模生产API中的已知UTE和潜在漏洞。这些结果强调了GraphQLer通过自动化、上下文感知漏洞检测主动保护GraphQL API的实用性。



## **30. GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms**

GraphAttack：利用LLM安全机制中的代表性盲点 cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13052v1) [paper-pdf](http://arxiv.org/pdf/2504.13052v1)

**Authors**: Sinan He, An Wang

**Abstract**: Large Language Models (LLMs) have been equipped with safety mechanisms to prevent harmful outputs, but these guardrails can often be bypassed through "jailbreak" prompts. This paper introduces a novel graph-based approach to systematically generate jailbreak prompts through semantic transformations. We represent malicious prompts as nodes in a graph structure with edges denoting different transformations, leveraging Abstract Meaning Representation (AMR) and Resource Description Framework (RDF) to parse user goals into semantic components that can be manipulated to evade safety filters. We demonstrate a particularly effective exploitation vector by instructing LLMs to generate code that realizes the intent described in these semantic graphs, achieving success rates of up to 87% against leading commercial LLMs. Our analysis reveals that contextual framing and abstraction are particularly effective at circumventing safety measures, highlighting critical gaps in current safety alignment techniques that focus primarily on surface-level patterns. These findings provide insights for developing more robust safeguards against structured semantic attacks. Our research contributes both a theoretical framework and practical methodology for systematically stress-testing LLM safety mechanisms.

摘要: 大型语言模型（LLM）配备了安全机制来防止有害输出，但这些护栏通常可以通过“越狱”提示绕过。本文介绍了一种新颖的基于图形的方法，通过语义转换系统地生成越狱提示。我们将恶意提示表示为图结构中的节点，边缘表示不同的转换，利用抽象意义表示（MRC）和资源描述框架（RDF）将用户目标解析为可以操纵以规避安全过滤器的语义组件。我们通过指示LLM生成实现这些语义图中描述的意图的代码来展示一种特别有效的利用载体，与领先的商业LLM相比，成功率高达87%。我们的分析表明，上下文框架和抽象在规避安全措施方面特别有效，凸显了当前主要关注表面模式的安全对齐技术中的关键差距。这些发现为开发针对结构化语义攻击的更强大的保护措施提供了见解。我们的研究为系统性压力测试LLM安全机制提供了理论框架和实践方法论。



## **31. From Sands to Mansions: Towards Automated Cyberattack Emulation with Classical Planning and Large Language Models**

从金沙到豪宅：利用经典规划和大型语言模型实现自动网络攻击模拟 cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2407.16928v3) [paper-pdf](http://arxiv.org/pdf/2407.16928v3)

**Authors**: Lingzhi Wang, Zhenyuan Li, Yi Jiang, Zhengkai Wang, Zonghan Guo, Jiahui Wang, Yangyang Wei, Xiangmin Shen, Wei Ruan, Yan Chen

**Abstract**: As attackers continually advance their tools, skills, and techniques during cyberattacks - particularly in modern Advanced Persistence Threats (APT) campaigns - there is a pressing need for a comprehensive and up-to-date cyberattack dataset to support threat-informed defense and enable benchmarking of defense systems in both academia and commercial solutions. However, there is a noticeable scarcity of cyberattack datasets: recent academic studies continue to rely on outdated benchmarks, while cyberattack emulation in industry remains limited due to the significant human effort and expertise required. Creating datasets by emulating advanced cyberattacks presents several challenges, such as limited coverage of attack techniques, the complexity of chaining multiple attack steps, and the difficulty of realistically mimicking actual threat groups. In this paper, we introduce modularized Attack Action and Attack Action Linking Model as a structured way to organizing and chaining individual attack steps into multi-step cyberattacks. Building on this, we propose Aurora, a system that autonomously emulates cyberattacks using third-party attack tools and threat intelligence reports with the help of classical planning and large language models. Aurora can automatically generate detailed attack plans, set up emulation environments, and semi-automatically execute the attacks. We utilize Aurora to create a dataset containing over 1,000 attack chains. To our best knowledge, Aurora is the only system capable of automatically constructing such a large-scale cyberattack dataset with corresponding attack execution scripts and environments. Our evaluation further demonstrates that Aurora outperforms the previous similar work and even the most advanced generative AI models in cyberattack emulation. To support further research, we published the cyberattack dataset and will publish the source code of Aurora.

摘要: 随着攻击者在网络攻击期间不断改进他们的工具、技能和技术--特别是在现代高级持久性威胁（APT）活动中--迫切需要一个全面且最新的网络攻击数据集来支持基于威胁的防御，并实现学术界和商业解决方案中的防御系统基准测试。然而，网络攻击数据集明显稀缺：最近的学术研究继续依赖过时的基准，而由于需要大量的人力和专业知识，行业中的网络攻击模拟仍然有限。通过模拟高级网络攻击创建数据集带来了几个挑战，例如攻击技术的覆盖范围有限、链接多个攻击步骤的复杂性以及真实模拟实际威胁组的困难。本文中，我们引入了模块化的攻击动作和攻击动作链接模型，作为一种将各个攻击步骤组织和链接为多步骤网络攻击的结构化方法。在此基础上，我们提出了Aurora，这是一个在经典规划和大型语言模型的帮助下使用第三方攻击工具和威胁情报报告自主模拟网络攻击的系统。Aurora可以自动生成详细的攻击计划、设置模拟环境并半自动执行攻击。我们利用Aurora创建一个包含1，000多个攻击链的数据集。据我们所知，Aurora是唯一能够自动构建如此大规模网络攻击数据集以及相应的攻击执行脚本和环境的系统。我们的评估进一步表明，Aurora在网络攻击模拟方面的表现优于之前的类似工作，甚至优于最先进的生成式人工智能模型。为了支持进一步的研究，我们发布了网络攻击数据集，并将发布Aurora的源代码。



## **32. ControlNET: A Firewall for RAG-based LLM System**

Control NET：基于RAG的LLM系统的防火墙 cs.CR

Project Page: https://ai.zjuicsr.cn/firewall

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.09593v2) [paper-pdf](http://arxiv.org/pdf/2504.09593v2)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.

摘要: 检索增强生成（RAG）显着增强了大型语言模型（LLM）的事实准确性和领域适应性。这一进步使它们能够在医疗保健、金融和企业应用程序等敏感领域广泛部署。RAG通过整合外部知识来缓解幻觉，但也会带来隐私风险和安全风险，尤其是数据泄露风险和数据中毒风险。虽然最近的研究探索了即时注射和中毒攻击，但在控制入站和出站查询流以减轻这些威胁的全面研究方面仍然存在显着差距。在本文中，我们提出了一种人工智能防火墙Controller NET，旨在保护基于RAG的LLM系统免受这些漏洞的影响。ControlNET通过利用激活转变现象来检测对抗性查询并通过语义分歧减轻其影响来控制查询流。我们使用最先进的开源LLM（Llama 3、Vicuna和Mistral）对四个不同的基准数据集（包括Mmarco、HotpotQA、FinQA和MedalSys）进行全面实验。我们的结果表明，ControlNET在检测和缓解安全威胁同时保持系统无害性方面达到了超过0.909 AUROC。总的来说，ControlNET提供了一种有效、健壮、无害的防御机制，标志着基于RAG的LLM系统安全部署的重大进步。



## **33. PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization**

PR攻击：通过二层优化对大型语言模型中的检索增强生成进行协调的预算-RAG攻击 cs.CR

Accepted at SIGIR 2025

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.07717v2) [paper-pdf](http://arxiv.org/pdf/2504.07717v2)

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods.

摘要: 大型语言模型（LLM）已在广泛的应用程序中表现出出色的性能，例如，医学问答、数学科学和代码生成。然而，它们也表现出固有的局限性，例如过时的知识和幻觉的易感性。检索增强一代（RAG）已成为解决这些问题的一个有希望的范式，但它也引入了新的漏洞。最近的工作重点是基于RAG的LLM的安全性，但现有的攻击方法面临三个关键挑战：（1）当只能将有限数量的有毒文本注入知识数据库时，它们的有效性急剧下降，（2）它们缺乏足够的隐蔽性，因为异常检测系统通常可以检测到攻击，这损害了它们的有效性，（3）它们依赖启发式方法来生成有毒文本，缺乏正式的优化框架和理论保证，这限制了它们的有效性和适用性。为了解决这些问题，我们提出了协调的预算-RAG攻击（PR-攻击），这是一种新型的优化驱动攻击，它将少量有毒文本引入知识数据库，同时在提示内嵌入后门触发器。激活时，触发器会使LLM生成对目标查询的预先设计的响应，同时在其他上下文中保持正常行为。这确保了高效率和隐形性。我们将攻击生成过程制定为一个双层优化问题，利用有原则的优化框架来开发最佳的有毒文本和触发器。跨不同LLM和数据集的广泛实验证明了PR-Attack的有效性，即使在数量有限的有毒文本的情况下也能实现很高的攻击成功率，并且与现有方法相比，隐身性显着提高。



## **34. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

LLM护栏中的快速注射和越狱检测 cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.

摘要: 大型语言模型（LLM）护栏系统旨在防止即时注入和越狱攻击。然而，他们仍然容易受到逃避技术的影响。我们演示了两种通过传统的字符注入方法和算法对抗机器学习（ML）规避技术绕过LLM提示注入和越狱检测系统的方法。通过对六种主要保护系统（包括微软的Azure Promise Shield和Meta的Promise Guard）进行测试，我们表明这两种方法都可以用来逃避检测，同时保持对抗性效用，在某些情况下实现高达100%的逃避成功。此外，我们还证明，对手可以通过利用离线白盒模型计算的单词重要性排名来提高针对黑盒目标的攻击成功率（ASB）。我们的研究结果揭示了当前LLM保护机制中的漏洞，并强调了对更坚固的护栏系统的需求。



## **35. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

软提示威胁：通过嵌入空间攻击开源LLM中的安全一致和取消学习 cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.

摘要: 当前对LLM对抗鲁棒性的研究重点是自然语言空间中的离散输入操纵，其可以直接转移到闭源模型。然而，这种方法忽视了开源模型的稳定发展。随着开源模型功能的进步，确保其安全性也变得越来越重要。然而，针对利用完整模型访问的开源LLM量身定制的攻击在很大程度上仍然未被探索。我们解决了这一研究空白并提出了嵌入空间攻击，该攻击直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击比离散攻击或模型微调更有效地规避模型对齐并触发有害行为。此外，我们在取消学习的背景下提出了一种新颖的威胁模型，并表明嵌入空间攻击可以从多个数据集和模型中未学习的LLM中提取据称已删除的信息。我们的研究结果强调将空间攻击嵌入到开源LLM中作为重要威胁模型。触发警告：附录包含LLM生成的带有暴力和骚扰的文本。



## **36. LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks**

LLM取消学习揭示了当前基准中强于预期的核心集效应 cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.10185v2) [paper-pdf](http://arxiv.org/pdf/2504.10185v2)

**Authors**: Soumyadeep Pal, Changsheng Wang, James Diffenderfer, Bhavya Kailkhura, Sijia Liu

**Abstract**: Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset.

摘要: 大型语言模型取消学习已成为通过从预训练模型中消除不希望的数据模型影响同时保持通用性来确保安全性和受控模型行为的一个关键挑战。最近做出了重大努力，致力于开发LLM忘记学习基准，例如WMDP（大规模杀伤性武器代理）和MUSE（机器忘记学习六路评估），促进标准化忘记学习性能评估和方法比较。尽管它们很有用，但我们首次在这些基准中发现了一种新颖的核心重置效应。具体来说，我们发现用原始（完整）忘记集实现的LLM取消学习可以使用明显较小的子集（充当“核心集”）有效地维护，例如，即使是随机选择，也只有忘记集的5%。这表明，即使在数据量极低的情况下，这些基准中的LLM取消学习也可以非常容易地执行。我们证明，无论使用何种LLM取消学习方法，例如NPO（负偏好优化）和RMU（代表误导取消学习），这种核心重置效应仍然很强，这些基准中流行的方法。令人惊讶的强烈核心集效应在各种数据选择方法中也很强大，从随机选择到更复杂的启发式方法。我们通过基于关键词的角度解释了LLM取消学习中的核心重置效应，表明仅从忘记集中提取的关键词对取消学习有效性做出了显着贡献，并表明当前的取消学习是由一组紧凑的高影响力令牌驱动的，而不是整个数据集。我们从其他维度（例如模式连接性和对越狱攻击的鲁棒性）进一步证明了未学习核心集的模型的忠实性。代码可访问https://github.com/OPTML-Group/MU-Coreset。



## **37. Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation**

LLM的信息引导水印：用于稳健且可追溯的文本生成的测试时框架 cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12108v1) [paper-pdf](http://arxiv.org/pdf/2504.12108v1)

**Authors**: Shizhan Cai, Liang Ding, Dacheng Tao

**Abstract**: The rapid development of Large Language Models (LLMs) has intensified concerns about content traceability and potential misuse. Existing watermarking schemes for sampled text often face trade-offs between maintaining text quality and ensuring robust detection against various attacks. To address these issues, we propose a novel watermarking scheme that improves both detectability and text quality by introducing a cumulative watermark entropy threshold. Our approach is compatible with and generalizes existing sampling functions, enhancing adaptability. Experimental results across multiple LLMs show that our scheme significantly outperforms existing methods, achieving over 80\% improvements on widely-used datasets, e.g., MATH and GSM8K, while maintaining high detection accuracy.

摘要: 大型语言模型（LLM）的快速发展加剧了人们对内容可追溯性和潜在滥用的担忧。现有的采样文本水印方案经常面临保持文本质量和确保针对各种攻击的鲁棒检测之间的权衡。为了解决这些问题，我们提出了一种新颖的水印方案，通过引入累积水印信息阈值来提高可检测性和文本质量。我们的方法与现有的采样功能兼容并推广，增强了适应性。多个LLM的实验结果表明，我们的方案显着优于现有方法，在广泛使用的数据集上实现了超过80%的改进，例如MATH和GSM 8 K，同时保持高检测准确性。



## **38. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

代理安全工作台（ASB）：对基于LLM的代理中的攻击和防御进行形式化和基准化 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.02644v3) [paper-pdf](http://arxiv.org/pdf/2410.02644v3)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.

摘要: 尽管基于LLM的代理在大型语言模型（LLM）的支持下可以使用外部工具和内存机制来解决复杂的现实世界任务，但它们也可能引入关键的安全漏洞。然而，现有文献并未全面评估针对基于LLM的代理的攻击和防御。为了解决这个问题，我们引入了代理安全工作台（ASB），这是一个全面的框架，旨在形式化、基准化和评估基于LLM的代理的攻击和防御，包括10种场景（例如，电子商务、自动驾驶、金融）、10个针对场景的代理、400多种工具、27种不同类型的攻击/防御方法和7个评估指标。基于ASB，我们对10种提示注入攻击、一种记忆中毒攻击、一种新颖的思想计划后门攻击、4种混合攻击以及13个LLM主干上的11种相应防御进行了基准测试。我们的基准测试结果揭示了代理操作不同阶段的关键漏洞，包括系统提示、用户提示处理、工具使用和内存检索，平均攻击成功率最高，为84.30%，但当前防御中表现出的有效性有限，揭示了社区在代理安全方面需要做的重要工作。我们还引入了一个新的指标来评估代理平衡实用性和安全性的能力。我们的代码可在https://github.com/agiresearch/ASB上找到。



## **39. On the Feasibility of Using MultiModal LLMs to Execute AR Social Engineering Attacks**

使用多模式LLM执行AR社会工程攻击的可行性 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.13209v1) [paper-pdf](http://arxiv.org/pdf/2504.13209v1)

**Authors**: Ting Bi, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Jun Zhang, Zui Tao, Kailong Wang, Liting Zhou, Yang Yang, Tianlong Yu

**Abstract**: Augmented Reality (AR) and Multimodal Large Language Models (LLMs) are rapidly evolving, providing unprecedented capabilities for human-computer interaction. However, their integration introduces a new attack surface for social engineering. In this paper, we systematically investigate the feasibility of orchestrating AR-driven Social Engineering attacks using Multimodal LLM for the first time, via our proposed SEAR framework, which operates through three key phases: (1) AR-based social context synthesis, which fuses Multimodal inputs (visual, auditory and environmental cues); (2) role-based Multimodal RAG (Retrieval-Augmented Generation), which dynamically retrieves and integrates contextual data while preserving character differentiation; and (3) ReInteract social engineering agents, which execute adaptive multiphase attack strategies through inference interaction loops. To verify SEAR, we conducted an IRB-approved study with 60 participants in three experimental configurations (unassisted, AR+LLM, and full SEAR pipeline) compiling a new dataset of 180 annotated conversations in simulated social scenarios. Our results show that SEAR is highly effective at eliciting high-risk behaviors (e.g., 93.3% of participants susceptible to email phishing). The framework was particularly effective in building trust, with 85% of targets willing to accept an attacker's call after an interaction. Also, we identified notable limitations such as ``occasionally artificial'' due to perceived authenticity gaps. This work provides proof-of-concept for AR-LLM driven social engineering attacks and insights for developing defensive countermeasures against next-generation augmented reality threats.

摘要: 增强现实（AR）和多模式大型语言模型（LLM）正在迅速发展，为人机交互提供了前所未有的能力。然而，它们的集成为社会工程引入了新的攻击面。本文通过我们提出的SEAR框架，首次系统地研究了使用多模式LLM策划AR驱动的社会工程攻击的可行性，该框架通过三个关键阶段运行：（1）基于AR的社会上下文合成，融合了多模式输入（视觉、听觉和环境线索）;（2）基于角色的多模式RAG（检索-增强代），动态检索和集成上下文数据，同时保留字符差异;和（3）ReInteract社会工程代理，通过推理交互循环执行自适应多阶段攻击策略。为了验证SEAR，我们对60名参与者进行了一项获得IRC批准的研究，参与者分为三种实验配置（无辅助、AR+LLM和完整SEAR管道），在模拟社交场景中编制了一个包含180个注释对话的新数据集。我们的结果表明，SEAR在引发高风险行为（例如，93.3%的参与者容易受到电子邮件网络钓鱼）。该框架在建立信任方面特别有效，85%的目标愿意在互动后接受攻击者的电话。此外，我们还发现了由于感知到的真实性差距而存在的显着局限性，例如“偶尔是人为的”。这项工作为AR-LLM驱动的社会工程攻击提供了概念验证，并为开发针对下一代增强现实威胁的防御对策提供了见解。



## **40. Progent: Programmable Privilege Control for LLM Agents**

Progent：LLM代理的可编程特权控制 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11703v1) [paper-pdf](http://arxiv.org/pdf/2504.11703v1)

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Linyu Wu, Hongwei Li, Wenbo Guo, Dawn Song

**Abstract**: LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility.   We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks.

摘要: LLM代理是人工智能系统的一种新兴形式，其中大型语言模型（LLM）作为中心组件，利用一组不同的工具来完成用户分配的任务。尽管LLM代理潜力巨大，但仍构成重大安全风险。在与外部世界互动时，他们可能会遇到攻击者的恶意命令，导致执行危险动作。解决这个问题的一个有希望的方法是执行最小特权原则：仅允许执行完成任务的必要动作，同时阻止不必要的动作。然而，实现这一点具有挑战性，因为它需要覆盖不同的代理场景，同时保持安全性和实用性。   我们引入Progent，这是LLM代理的第一个特权控制机制。其核心是一种特定于领域的语言，用于灵活表达代理执行期间应用的特权控制策略。这些策略为工具调用提供了细粒度的约束，决定何时允许工具调用，并在不允许时指定后备。这使代理开发人员和用户能够为他们的特定用例制定合适的策略，并确定性地实施这些策略以保证安全性。由于其模块化设计，集成Progent不会改变代理的内部结构，只需要对代理的实现进行最小的更改，从而增强了其实用性和广泛采用的潜力。为了自动化策略编写，我们利用LLM根据用户查询生成策略，然后动态更新以提高安全性和实用性。我们的广泛评估表明，它可以实现强大的安全性，同时在三个不同的场景或基准测试中保持高实用性：AgentDojo，ASB和AgentPoison。此外，我们还进行了深入的分析，展示了其核心组件的有效性以及自动化策略生成针对自适应攻击的弹性。



## **41. Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction**

通过LLM辅助频谱图的“错别字”纠正，使对噪音键盘的声学侧通道攻击变得可行 cs.CR

Length: 13 pages Figures: 5 figures Tables: 7 tables Keywords:  Acoustic side-channel attacks, machine learning, Visual Transformers, Large  Language Models (LLMs), security Conference: Accepted at the 19th USENIX WOOT  Conference on Offensive Technologies (WOOT '25). Licensing: This paper is  submitted under the CC BY Creative Commons Attribution license. arXiv admin  note: text overlap with arXiv:2502.09782

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11622v1) [paper-pdf](http://arxiv.org/pdf/2504.11622v1)

**Authors**: Seyyed Ali Ayati, Jin Hyun Park, Yichen Cai, Marcus Botacin

**Abstract**: The large integration of microphones into devices increases the opportunities for Acoustic Side-Channel Attacks (ASCAs), as these can be used to capture keystrokes' audio signals that might reveal sensitive information. However, the current State-Of-The-Art (SOTA) models for ASCAs, including Convolutional Neural Networks (CNNs) and hybrid models, such as CoAtNet, still exhibit limited robustness under realistic noisy conditions. Solving this problem requires either: (i) an increased model's capacity to infer contextual information from longer sequences, allowing the model to learn that an initially noisily typed word is the same as a futurely collected non-noisy word, or (ii) an approach to fix misidentified information from the contexts, as one does not type random words, but the ones that best fit the conversation context. In this paper, we demonstrate that both strategies are viable and complementary solutions for making ASCAs practical. We observed that no existing solution leverages advanced transformer architectures' power for these tasks and propose that: (i) Visual Transformers (VTs) are the candidate solutions for capturing long-term contextual information and (ii) transformer-powered Large Language Models (LLMs) are the candidate solutions to fix the ``typos'' (mispredictions) the model might make. Thus, we here present the first-of-its-kind approach that integrates VTs and LLMs for ASCAs.   We first show that VTs achieve SOTA performance in classifying keystrokes when compared to the previous CNN benchmark. Second, we demonstrate that LLMs can mitigate the impact of real-world noise. Evaluations on the natural sentences revealed that: (i) incorporating LLMs (e.g., GPT-4o) in our ASCA pipeline boosts the performance of error-correction tasks; and (ii) the comparable performance can be attained by a lightweight, fine-tuned smaller LLM (67 times smaller than GPT-4o), using...

摘要: 麦克风大量集成到设备中增加了声学侧道攻击（ASCA）的机会，因为这些攻击可用于捕获可能泄露敏感信息的麦克风音频信号。然而，当前ASCA的最新技术水平（SOTA）模型（包括卷积神经网络（CNN）和混合模型（例如CoAtNet））在现实噪音条件下仍然表现出有限的鲁棒性。解决这个问题需要：（i）提高模型从更长的序列中推断上下文信息的能力，允许模型学习最初输入的有噪音的单词与未来收集的无噪音单词相同，或者（ii）修复来自上下文的错误识别信息的方法，因为输入的不是随机单词，而是最适合对话上下文的单词。在本文中，我们证明了这两种策略都是使ASCA实用的可行且相辅相成的解决方案。我们观察到，没有现有的解决方案利用高级Transformer架构的能力来完成这些任务，并建议：（i）视觉转换器（VT）是捕获长期上下文信息的候选解决方案，（ii）转换器驱动的大型语言模型（LLM）是修复模型可能造成的“拼写错误”（预测错误）的候选解决方案。因此，我们在这里介绍了一种首创的方法，该方法将VT和LLM集成到ASCA中。   我们首先表明，与之前的CNN基准相比，VT在分类击键方面实现了SOTA性能。其次，我们证明LLM可以减轻现实世界噪音的影响。对自然句子的评估显示：（i）纳入LLM（例如，我们的ASCA管道中的GPT-4 o）提高了错误纠正任务的性能;并且（ii）通过轻量级、微调的较小LLM（比GPT-4 o小67倍）可以获得相当的性能，使用.



## **42. Propaganda via AI? A Study on Semantic Backdoors in Large Language Models**

通过人工智能进行宣传？大型语言模型中的语义后门研究 cs.CL

18 pages, 1 figure

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.12344v1) [paper-pdf](http://arxiv.org/pdf/2504.12344v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across myriad language tasks, yet they remain vulnerable to backdoor attacks, where adversaries implant hidden triggers that systematically manipulate model outputs. Traditional defenses focus on explicit token-level anomalies and therefore overlook semantic backdoors-covert triggers embedded at the conceptual level (e.g., ideological stances or cultural references) that rely on meaning-based cues rather than lexical oddities. We first show, in a controlled finetuning setting, that such semantic backdoors can be implanted with only a small poisoned corpus, establishing their practical feasibility. We then formalize the notion of semantic backdoors in LLMs and introduce a black-box detection framework, RAVEN (short for "Response Anomaly Vigilance for uncovering semantic backdoors"), which combines semantic entropy with cross-model consistency analysis. The framework probes multiple models with structured topic-perspective prompts, clusters the sampled responses via bidirectional entailment, and flags anomalously uniform outputs; cross-model comparison isolates model-specific anomalies from corpus-wide biases. Empirical evaluations across diverse LLM families (GPT-4o, Llama, DeepSeek, Mistral) uncover previously undetected semantic backdoors, providing the first proof-of-concept evidence of these hidden vulnerabilities and underscoring the urgent need for concept-level auditing of deployed language models. We open-source our code and data at https://github.com/NayMyatMin/RAVEN.

摘要: 大型语言模型（LLM）在无数语言任务中表现出出色的性能，但它们仍然容易受到后门攻击，即对手植入隐藏触发器来系统性地操纵模型输出。传统防御专注于显式标记级异常，因此忽视了嵌入在概念级的语义后门隐蔽触发器（例如，意识形态立场或文化参考）依赖于基于意义的线索，而不是词汇上的怪异。我们首先表明，在受控微调环境中，这种语义后门只能植入一个小的有毒主体，从而建立了它们的实际可行性。然后，我们在LLM中形式化了语义后门的概念，并引入了黑匣子检测框架RAVEN（“揭露语义后门的响应异常警戒”的缩写），该框架将语义熵与跨模型一致性分析相结合。该框架通过结构化的主题视角提示来探索多个模型，通过双向蕴含对采样的响应进行聚集，并标记出极其均匀的输出;跨模型比较将模型特定的异常与整个群体的偏差隔离开来。对不同LLM家族（GPT-4 o、Llama、DeepSeek、Mistral）的经验评估揭示了之前未检测到的语义后门，提供了这些隐藏漏洞的第一个概念验证证据，并强调了对已部署语言模型进行概念级审计的迫切需要。我们在https://github.com/NayMyatMin/RAVEN上开源我们的代码和数据。



## **43. Lateral Phishing With Large Language Models: A Large Organization Comparative Study**

大型语言模型的横向网络钓鱼：大型组织比较研究 cs.CR

Accepted for publication in IEEE Access. This version includes  revisions following peer review

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2401.09727v2) [paper-pdf](http://arxiv.org/pdf/2401.09727v2)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nicole Beebe, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The emergence of Large Language Models (LLMs) has heightened the threat of phishing emails by enabling the generation of highly targeted, personalized, and automated attacks. Traditionally, many phishing emails have been characterized by typos, errors, and poor language. These errors can be mitigated by LLMs, potentially lowering the barrier for attackers. Despite this, there is a lack of large-scale studies comparing the effectiveness of LLM-generated lateral phishing emails to those crafted by humans. Current literature does not adequately address the comparative effectiveness of LLM and human-generated lateral phishing emails in a real-world, large-scale organizational setting, especially considering the potential for LLMs to generate more convincing and error-free phishing content. To address this gap, we conducted a pioneering study within a large university, targeting its workforce of approximately 9,000 individuals including faculty, staff, administrators, and student workers. Our results indicate that LLM-generated lateral phishing emails are as effective as those written by communications professionals, emphasizing the critical threat posed by LLMs in leading phishing campaigns. We break down the results of the overall phishing experiment, comparing vulnerability between departments and job roles. Furthermore, to gather qualitative data, we administered a detailed questionnaire, revealing insights into the reasons and motivations behind vulnerable employee's actions. This study contributes to the understanding of cyber security threats in educational institutions and provides a comprehensive comparison of LLM and human-generated phishing emails' effectiveness, considering the potential for LLMs to generate more convincing content. The findings highlight the need for enhanced user education and system defenses to mitigate the growing threat of AI-powered phishing attacks.

摘要: 大型语言模型（LLM）的出现通过生成高度针对性、个性化和自动化的攻击，加剧了网络钓鱼电子邮件的威胁。传统上，许多网络钓鱼电子邮件的特点是拼写错误、错误和语言拙劣。这些错误可以通过LLM来缓解，从而可能降低攻击者的障碍。尽管如此，缺乏大规模研究将LLM生成的横向网络钓鱼电子邮件与人类制作的横向网络钓鱼电子邮件的有效性进行比较。当前的文献没有充分解决LLM和人类生成的横向网络钓鱼电子邮件在现实世界、大规模组织环境中的比较有效性，特别是考虑到LLM生成更令人信服且无错误的网络钓鱼内容的潜力。为了解决这一差距，我们在一所大型大学内进行了一项开创性研究，目标是其约9，000名员工，包括教职员工、管理人员和学生工作者。我们的结果表明，LLM生成的横向网络钓鱼电子邮件与通信专业人士撰写的电子邮件一样有效，强调了LLM在领先的网络钓鱼活动中构成的严重威胁。我们分解了整个网络钓鱼实验的结果，比较了部门和工作角色之间的脆弱性。此外，为了收集定性数据，我们进行了一份详细的调查问卷，揭示了对弱势员工行为背后的原因和动机的见解。这项研究有助于了解教育机构的网络安全威胁，并对LLM和人类生成的网络钓鱼电子邮件的有效性进行了全面比较，同时考虑到LLM生成更令人信服的内容的潜力。研究结果凸显了加强用户教育和系统防御的必要性，以减轻人工智能驱动的网络钓鱼攻击日益严重的威胁。



## **44. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统很容易受到对抗攻击。不受限制的对抗攻击通常操纵图像的语义内容（例如，颜色或纹理）来创建既有效又逼真的对抗示例。最近的作品利用扩散倒置过程将图像映射到潜在空间，其中通过引入扰动来操纵高级语义。然而，它们通常会导致去噪输出中出现严重的语义扭曲，并且效率低下。在这项研究中，我们提出了一种名为语义一致的无限制对抗攻击（SCA）的新型框架，该框架采用倒置方法来提取编辑友好的噪音图，并利用多模式大型语言模型（MLLM）在整个过程中提供语义指导。在MLLM提供丰富的语义信息的情况下，我们使用一系列编辑友好的噪音图来执行每一步的DDPM去噪过程，并利用DeliverSolver ++加速这一过程，实现具有语义一致性的高效采样。与现有方法相比，我们的框架能够高效生成表现出最小可辨别的语义变化的对抗性示例。因此，我们首次引入语义一致的对抗示例（SCAE）。大量的实验和可视化已经证明了SCA的高效率，特别是平均比最先进的攻击快12倍。本文的研究可以进一步引起人们对多媒体信息安全的关注。



## **45. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

显而易见的不可见威胁：LLM-Powered GUI代理对Fine-Print注入的脆弱性 cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.

摘要: 由大型语言模型（LLM）驱动的图形用户界面代理是一个专门的自治系统，根据高级指令代表用户执行任务。它通过感知和解释相关应用程序的图形用户界面（GUIs）（通常是视觉上的），推断必要的操作序列，然后通过执行单击、打字和点击等操作与GUIs交互来实现这一目标。为了完成现实世界的任务，例如填写表格或预订服务，图形用户界面代理通常需要处理和处理敏感用户数据。然而，这种自主性带来了新的隐私和安全风险。对手可以将恶意内容注入图形用户界面，从而改变代理行为或导致私人信息的意外泄露。这些攻击通常利用代理和人类用户的视觉显著性之间的差异，或者代理检测任务自动化中上下文完整性违规的能力有限。在本文中，我们描述了六种类型的此类攻击，并进行了一项实验研究，使用六个最先进的图形用户界面代理、234个对抗性网页和39名人类参与者来测试这些攻击。我们的研究结果表明，图形用户界面代理非常容易受到攻击，特别是对于上下文嵌入式威胁。此外，人类用户也容易受到许多此类攻击，这表明简单的人类监督可能无法可靠地防止故障。这种错位凸显了隐私感知代理设计的必要性。我们提出了实用的防御策略，为开发更安全、更可靠的图形用户界面代理提供信息。



## **46. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

探索LLM授权建议的后门攻击和防御 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.

摘要: 大型语言模型（LLM）与推荐系统（RecSys）的融合极大地提高了个性化推荐并引起了广泛关注。尽管取得了令人印象深刻的进展，但基于LLM的RecSys抵御后门攻击的安全性在很大程度上仍然没有得到充分的探索。在本文中，我们提出了一个新问题：具有特定触发器的后门是否会被注入到基于LLM的Recsys中，从而导致当后门触发器附加到项目标题时推荐响应的操纵？为了调查基于LLM的RecSys在后门攻击下的漏洞，我们提出了一种新的攻击框架，称为RecSys后门注入中毒（BadRec）。BadRec通过触发器扰乱这些物品的标题，并雇用几名虚假用户与这些物品互动，有效地毒害了训练集，并为基于LLM的RecSys注入后门。全面的实验表明，仅用对抗性示例毒害1%的训练数据就足以成功植入后门，从而能够操纵推荐。为了进一步减轻此类安全威胁，我们提出了一种名为毒药扫描仪（P-Scanner）的通用防御策略。具体来说，我们引入了基于LLM的毒物扫描仪，通过利用LLM强大的语言理解能力和丰富的知识来检测有毒物品。触发增强代理被用来生成不同的合成触发器，以引导中毒扫描器学习中毒物品检测任务的特定于领域的知识。在三个真实数据集上的大量实验验证了所提出的P-Scanner的有效性。



## **47. QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models**

QAVA：对大型视觉语言模型的查询不可知视觉攻击 cs.CV

Accepted by NAACL 2025 main

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11038v1) [paper-pdf](http://arxiv.org/pdf/2504.11038v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at https://github.com/btzyd/qava.

摘要: 在典型的多模式任务中，例如视觉问题解答（VQA），针对特定图像和问题的对抗攻击可能会导致大型视觉语言模型（LVLM）提供错误的答案。然而，单个图像与多个问题关联是常见的，即使对于受到特定问题攻击的对抗图像，LVLM仍然可以正确回答其他问题。为了解决这个问题，我们引入了查询不可知视觉攻击（QAVA），其目的是创建强大的对抗性示例，这些示例会对未指定和未知的问题生成错误的响应。与针对特定图像和问题的传统对抗攻击相比，QAVA显着增强了问题未知时图像攻击的有效性和效率，实现了与针对已知目标问题的攻击相当的性能。我们的研究扩大了实际环境中对LVLM的视觉对抗攻击的范围，揭示了以前被忽视的漏洞，特别是在视觉对抗威胁的背景下。该代码可在https://github.com/btzyd/qava上获取。



## **48. Concept Enhancement Engineering: A Lightweight and Efficient Robust Defense Against Jailbreak Attacks in Embodied AI**

概念增强工程：针对智能人工智能越狱攻击的轻量级、高效的稳健防御 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.13201v1) [paper-pdf](http://arxiv.org/pdf/2504.13201v1)

**Authors**: Jirui Yang, Zheyu Lin, Shuhan Yang, Zhihui Lu, Xin Du

**Abstract**: Embodied Intelligence (EI) systems integrated with large language models (LLMs) face significant security risks, particularly from jailbreak attacks that manipulate models into generating harmful outputs or executing unsafe physical actions. Traditional defense strategies, such as input filtering and output monitoring, often introduce high computational overhead or interfere with task performance in real-time embodied scenarios. To address these challenges, we propose Concept Enhancement Engineering (CEE), a novel defense framework that leverages representation engineering to enhance the safety of embodied LLMs by dynamically steering their internal activations. CEE operates by (1) extracting multilingual safety patterns from model activations, (2) constructing control directions based on safety-aligned concept subspaces, and (3) applying subspace concept rotation to reinforce safe behavior during inference. Our experiments demonstrate that CEE effectively mitigates jailbreak attacks while maintaining task performance, outperforming existing defense methods in both robustness and efficiency. This work contributes a scalable and interpretable safety mechanism for embodied AI, bridging the gap between theoretical representation engineering and practical security applications. Our findings highlight the potential of latent-space interventions as a viable defense paradigm against emerging adversarial threats in physically grounded AI systems.

摘要: 与大型语言模型（LLM）集成的分布式智能（EI）系统面临着重大的安全风险，特别是来自操纵模型生成有害输出或执行不安全物理动作的越狱攻击。传统的防御策略，例如输入过滤和输出监控，通常会引入高计算负担或干扰实时具体场景中的任务性能。为了应对这些挑战，我们提出了概念增强工程（CEE），这是一种新型防御框架，它利用表示工程通过动态引导其内部激活来增强具体LLM的安全性。CEE的运作方式是：（1）从模型激活中提取多语言安全模式，（2）基于安全对齐的概念子空间构建控制方向，以及（3）应用子空间概念旋转来加强推理期间的安全行为。我们的实验表明，CEE有效地缓解了越狱攻击，同时保持了任务性能，在鲁棒性和效率方面都优于现有的防御方法。这项工作为嵌入式人工智能提供了可扩展和可解释的安全机制，弥合了理论表示工程和实际安全应用之间的差距。我们的研究结果强调了潜伏空间干预作为一种可行的防御范式的潜力，以应对物理基础人工智能系统中出现的对抗威胁。



## **49. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

迈向智能和安全的云：大型语言模型增强主动防御 cs.CR

7 pages; In submission

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.21051v2) [paper-pdf](http://arxiv.org/pdf/2412.21051v2)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods.

摘要: 云计算技术的快速发展和云应用程序数量的不断增加为日常生活带来了大量好处。然而，不同组件的多样性和复杂性对云安全构成了重大挑战，特别是在处理复杂和高级的网络攻击时。生成式基础模型（GFM）的最新进展，特别是大型语言模型（LLM），为安全智能提供了有前途的解决方案。通过利用语言理解、数据分析、任务推理、行动规划和代码生成方面的强大能力，我们提出了LLM-PD，这是一种新型的主动防御架构，可以以主动的方式击败各种威胁。LLM-PD可以通过全面的数据分析和顺序推理，以及在目标云上动态创建和部署可操作的防御机制来有效地做出决策。此外，它可以根据从之前的交互中学到的经验灵活地自我进化，并在无需额外训练的情况下适应新的攻击场景。实验结果证明了其在防御有效性和效率方面的出色能力，特别是与其他现有方法相比具有出色的成功率。



## **50. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性即时蒸馏 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.15244v2) [paper-pdf](http://arxiv.org/pdf/2411.15244v2)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.

摘要: 对比图像预训练（CLIP）等大型预训练视觉语言模型（VLM）已被证明容易受到对抗攻击，这引发了人们对其在自动驾驶和医疗诊断等安全关键应用中部署的担忧。对抗性提示调整（APT）是对预训练的VLM进行鲁棒化的一种有希望的方法，它在提示调整的过程中应用对抗性训练。然而，现有的APT方法大多是单模式方法，仅为视觉或文本模式设计提示，从而限制了其稳健性或清晰准确性的有效性。在这项工作中，我们提出了对抗性提示蒸馏（APT），这是一个双峰知识蒸馏框架，通过将APT与多模式知识转移集成来增强APT。APT优化视觉和文本模式的提示，同时从干净的预培训教师CLIP模型中提取知识。对多个基准数据集的广泛实验证明了我们的APT方法在对抗稳健性和精确性方面优于当前最先进的APT方法。APT的有效性也验证了使用非稳健教师来提高微调后的VLM的通用性和稳健性的可能性。



