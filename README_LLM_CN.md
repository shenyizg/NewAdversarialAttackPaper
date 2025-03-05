# Latest Large Language Model Attack Papers
**update at 2025-03-05 10:17:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LLM-Safety Evaluations Lack Robustness**

LLM-安全性评估缺乏稳健性 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02574v1) [paper-pdf](http://arxiv.org/pdf/2503.02574v1)

**Authors**: Tim Beyer, Sophie Xhonneux, Simon Geisler, Gauthier Gidel, Leo Schwinn, Stephan Günnemann

**Abstract**: In this paper, we argue that current safety alignment research efforts for large language models are hindered by many intertwined sources of noise, such as small datasets, methodological inconsistencies, and unreliable evaluation setups. This can, at times, make it impossible to evaluate and compare attacks and defenses fairly, thereby slowing progress. We systematically analyze the LLM safety evaluation pipeline, covering dataset curation, optimization strategies for automated red-teaming, response generation, and response evaluation using LLM judges. At each stage, we identify key issues and highlight their practical impact. We also propose a set of guidelines for reducing noise and bias in evaluations of future attack and defense papers. Lastly, we offer an opposing perspective, highlighting practical reasons for existing limitations. We believe that addressing the outlined problems in future research will improve the field's ability to generate easily comparable results and make measurable progress.

摘要: 在本文中，我们认为目前针对大型语言模型的安全对齐研究工作受到许多相互交织的噪声源的阻碍，如小数据集、方法不一致和不可靠的评估设置。这有时会使人们无法公平地评估和比较攻击和防御，从而减缓进展。我们系统地分析了LLM安全评估管道，包括数据集管理、自动红团队的优化策略、响应生成和使用LLM评判器的响应评估。在每个阶段，我们确定关键问题并强调其实际影响。我们还提出了一套指导方针，以减少未来攻击和防御论文评估中的噪音和偏见。最后，我们提供了一个相反的观点，强调了现有限制的实际原因。我们认为，在今后的研究中解决概述的问题将提高该领域产生容易比较的结果和取得可衡量的进展的能力。



## **2. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

TPIA：针对面向代码的大型语言模型的特定目标提示注入攻击 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2407.09164v5) [paper-pdf](http://arxiv.org/pdf/2407.09164v5)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully exploited to simplify and facilitate programming. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. In this paper, we reveal that both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process, which may not be practical; adversarial attacks struggle with fulfilling specific malicious purposes. To alleviate these problems, this paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an attack success rate of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an attack success rate of over 90%) in all threat cases, using only a 12-token non-functional perturbation.

摘要: 最近，面向代码的大型语言模型(Code LLM)已经被广泛并成功地利用来简化和促进编程。不幸的是，一些开创性的工作表明，这些代码LLM容易受到后门和对手的攻击。前者毒化训练数据或模型参数，在遇到触发器时劫持LLMS生成恶意代码片段。后者制作恶意敌意输入代码以降低生成代码的质量。在本文中，我们揭示了这两种攻击都有一些固有的局限性：后门攻击依赖于对手控制模型训练过程的能力，这可能是不实用的；对抗性攻击难以实现特定的恶意目的。针对这些问题，提出了一种新的针对代码LLMS的攻击范式，即目标特定的即时注入攻击(TPIA)。TPIA生成包含恶意指令信息的非功能性扰动，并通过将它们传播到可能使用的依赖项(例如，包或RAG的知识库)，将它们插入到受害者的代码上下文中。它诱导代码LLM在目标位置生成攻击者指定的恶意代码片段。一般而言，我们通过基于贪婪令牌搜索的对抗性优化将攻击者指定的恶意目标压缩为扰动。我们收集了13个具有代表性的恶意目标，为三种流行的编程语言设计了31个威胁案例。实验表明，在所有威胁情况下，仅使用12个令牌的非功能扰动，我们的TPIA就可以成功攻击三个典型的开源代码LLM(攻击成功率高达97.9%)和两个主流商业代码LLM集成应用(攻击成功率超过90%)。



## **3. Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents**

自适应攻击突破了对LLM代理间接即时注入攻击的防御 cs.CR

17 pages, 5 figures, 6 tables (NAACL 2025 Findings)

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.00061v2) [paper-pdf](http://arxiv.org/pdf/2503.00061v2)

**Authors**: Qiusi Zhan, Richard Fang, Henil Shalin Panchal, Daniel Kang

**Abstract**: Large Language Model (LLM) agents exhibit remarkable performance across diverse applications by using external tools to interact with environments. However, integrating external tools introduces security risks, such as indirect prompt injection (IPI) attacks. Despite defenses designed for IPI attacks, their robustness remains questionable due to insufficient testing against adaptive attacks. In this paper, we evaluate eight different defenses and bypass all of them using adaptive attacks, consistently achieving an attack success rate of over 50%. This reveals critical vulnerabilities in current defenses. Our research underscores the need for adaptive attack evaluation when designing defenses to ensure robustness and reliability. The code is available at https://github.com/uiuc-kang-lab/AdaptiveAttackAgent.

摘要: 大型语言模型（LLM）代理通过使用外部工具与环境交互，在不同的应用程序中表现出出色的性能。然而，集成外部工具会带来安全风险，例如间接提示注入（IPI）攻击。尽管针对IPI攻击设计了防御措施，但由于针对自适应攻击的测试不足，其稳健性仍然值得怀疑。在本文中，我们评估了八种不同的防御措施，并使用自适应攻击绕过了所有防御措施，始终实现了超过50%的攻击成功率。这揭示了当前防御系统中的关键漏洞。我们的研究强调了在设计防御以确保稳健性和可靠性时需要进行自适应攻击评估。该代码可在https://github.com/uiuc-kang-lab/AdaptiveAttackAgent上获取。



## **4. Confidential Prompting: Protecting User Prompts from Cloud LLM Providers**

机密预算：保护用户预算免受云LLM提供商的预算 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2409.19134v3) [paper-pdf](http://arxiv.org/pdf/2409.19134v3)

**Authors**: In Gim, Caihua Li, Lin Zhong

**Abstract**: Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring model confidentiality, output invariance, and compute efficiency. We introduce Secure Partitioned Decoding (SPD), which uses confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, Prompt Obfuscation (PO), to ensure robustness against reconstruction attacks on SPD. We demonstrate our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution enables privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.

摘要: 我们的工作解决了在云托管大型语言模型（LLM）服务中保护用户输入的挑战，同时确保模型机密性、输出不变性和计算效率。我们引入了安全分区解码（SPD），它使用机密计算将用户提示限制在可信执行环境（TEK），即机密虚拟机（CGM），同时允许服务提供商高效地生成令牌。我们还引入了一种新型加密方法--提示混淆（PO），以确保抵御SPD重建攻击的鲁棒性。我们证明我们的方法既保留了即时的保密性，又保留了LLM服务效率。我们的解决方案支持保护隐私的云LLM服务，可以处理敏感提示，例如临床记录、财务数据和个人信息。



## **5. De-identification is not enough: a comparison between de-identified and synthetic clinical notes**

去识别还不够：去识别和合成临床笔记之间的比较 cs.CL

https://www.nature.com/articles/s41598-024-81170-y

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2402.00179v2) [paper-pdf](http://arxiv.org/pdf/2402.00179v2)

**Authors**: Atiquer Rahman Sarkar, Yao-Shun Chuang, Noman Mohammed, Xiaoqian Jiang

**Abstract**: For sharing privacy-sensitive data, de-identification is commonly regarded as adequate for safeguarding privacy. Synthetic data is also being considered as a privacy-preserving alternative. Recent successes with numerical and tabular data generative models and the breakthroughs in large generative language models raise the question of whether synthetically generated clinical notes could be a viable alternative to real notes for research purposes. In this work, we demonstrated that (i) de-identification of real clinical notes does not protect records against a membership inference attack, (ii) proposed a novel approach to generate synthetic clinical notes using the current state-of-the-art large language models, (iii) evaluated the performance of the synthetically generated notes in a clinical domain task, and (iv) proposed a way to mount a membership inference attack where the target model is trained with synthetic data. We observed that when synthetically generated notes closely match the performance of real data, they also exhibit similar privacy concerns to the real data. Whether other approaches to synthetically generated clinical notes could offer better trade-offs and become a better alternative to sensitive real notes warrants further investigation.

摘要: 对于共享隐私敏感数据，消除身份识别通常被认为足以保护隐私。合成数据也被认为是一种保护隐私的选择。最近数字和表格数据生成模型的成功以及大型生成语言模型的突破提出了一个问题，即合成生成的临床笔记是否可以作为用于研究目的的真实笔记的可行替代方案。在这项工作中，我们证明了(I)真实临床笔记的去识别并不能保护记录免受成员关系推理攻击，(Ii)提出了一种使用当前最先进的大型语言模型生成合成临床笔记的新方法，(Iii)评估了合成生成的笔记在临床领域任务中的性能，以及(Iv)提出了一种利用合成数据训练目标模型的成员关系推理攻击的方法。我们观察到，当合成的笔记与真实数据的性能非常匹配时，它们也表现出与真实数据相似的隐私问题。合成临床笔记的其他方法是否可以提供更好的权衡，并成为敏感的真实笔记的更好替代方案，值得进一步研究。



## **6. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

通过大型语言模型越狱受保护的文本到图像模型 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01839v1) [paper-pdf](http://arxiv.org/pdf/2503.01839v1)

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.

摘要: 文本到图像模型可能会生成有害内容，例如色情图像，特别是在提交不安全提示时。为了解决这个问题，通常在文本到图像模型之上添加安全过滤器，或者对模型本身进行调整以减少有害输出。然而，当攻击者战略性地设计对抗提示来绕过这些安全护栏时，这些防御仍然容易受到攻击。在这项工作中，我们提出了ObjetTune，这是一种使用微调的大型语言模型来越狱具有安全护栏的文本到图像模型的方法。与其他需要对目标模型重复查询的基于查询的越狱攻击不同，我们的攻击在微调AttackLLM后有效地生成对抗提示。我们在三个不安全提示数据集和五个安全护栏上评估我们的方法。我们的结果表明，我们的方法有效地绕过了安全护栏，优于现有的无框攻击，并且还促进了其他基于查询的攻击。



## **7. AutoAdvExBench: Benchmarking autonomous exploitation of adversarial example defenses**

AutoAdvExBench：对抗性示例防御的自主利用基准 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01811v1) [paper-pdf](http://arxiv.org/pdf/2503.01811v1)

**Authors**: Nicholas Carlini, Javier Rando, Edoardo Debenedetti, Milad Nasr, Florian Tramèr

**Abstract**: We introduce AutoAdvExBench, a benchmark to evaluate if large language models (LLMs) can autonomously exploit defenses to adversarial examples. Unlike existing security benchmarks that often serve as proxies for real-world tasks, bench directly measures LLMs' success on tasks regularly performed by machine learning security experts. This approach offers a significant advantage: if a LLM could solve the challenges presented in bench, it would immediately present practical utility for adversarial machine learning researchers. We then design a strong agent that is capable of breaking 75% of CTF-like ("homework exercise") adversarial example defenses. However, we show that this agent is only able to succeed on 13% of the real-world defenses in our benchmark, indicating the large gap between difficulty in attacking "real" code, and CTF-like code. In contrast, a stronger LLM that can attack 21% of real defenses only succeeds on 54% of CTF-like defenses. We make this benchmark available at https://github.com/ethz-spylab/AutoAdvExBench.

摘要: 我们引入了AutoAdvExB边，这是一个基准，用来评估大型语言模型(LLM)是否能够自主地利用对对手例子的防御。与通常作为真实任务代理的现有安全基准不同，BASE直接衡量LLMS在机器学习安全专家定期执行的任务中的成功程度。这种方法提供了一个显著的优势：如果LLM能够解决BASE中提出的挑战，它将立即为对抗性机器学习研究人员提供实用价值。然后，我们设计了一个强大的代理，它能够打破75%的CTF类(“家庭作业练习”)对抗性范例防御。然而，我们表明，在我们的基准测试中，该代理只能够在13%的真实世界防御中成功，这表明攻击“真实”代码的难度与类似CTF的代码之间存在巨大差距。相比之下，更强大的LLM可以攻击21%的真实防御，只能在54%的CTF类防御上成功。我们在https://github.com/ethz-spylab/AutoAdvExBench.上提供此基准测试



## **8. Building Safe GenAI Applications: An End-to-End Overview of Red Teaming for Large Language Models**

构建安全的GenAI应用程序：大型语言模型红色团队的端到端概述 cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01742v1) [paper-pdf](http://arxiv.org/pdf/2503.01742v1)

**Authors**: Alberto Purpura, Sahil Wadhwa, Jesse Zymet, Akshay Gupta, Andy Luo, Melissa Kazemi Rad, Swapnil Shinde, Mohammad Shahed Sorower

**Abstract**: The rapid growth of Large Language Models (LLMs) presents significant privacy, security, and ethical concerns. While much research has proposed methods for defending LLM systems against misuse by malicious actors, researchers have recently complemented these efforts with an offensive approach that involves red teaming, i.e., proactively attacking LLMs with the purpose of identifying their vulnerabilities. This paper provides a concise and practical overview of the LLM red teaming literature, structured so as to describe a multi-component system end-to-end. To motivate red teaming we survey the initial safety needs of some high-profile LLMs, and then dive into the different components of a red teaming system as well as software packages for implementing them. We cover various attack methods, strategies for attack-success evaluation, metrics for assessing experiment outcomes, as well as a host of other considerations. Our survey will be useful for any reader who wants to rapidly obtain a grasp of the major red teaming concepts for their own use in practical applications.

摘要: 大型语言模型(LLM)的快速增长带来了重大的隐私、安全和伦理问题。虽然许多研究已经提出了保护LLM系统免受恶意行为者滥用的方法，但研究人员最近又用一种涉及红色团队的进攻性方法来补充这些努力，即主动攻击LLM，目的是识别它们的漏洞。本文提供了LLM红队文献的简明而实用的概述，其结构旨在描述端到端的多组件系统。为了激励红色团队，我们调查了一些备受瞩目的低成本管理系统的初始安全需求，然后深入研究红色团队系统的不同组件以及实施它们的软件包。我们涵盖了各种攻击方法、攻击成功评估策略、评估实验结果的指标以及许多其他考虑因素。我们的调查对于任何想要快速掌握主要的红色团队概念以便在实际应用中使用的读者都是有用的。



## **9. Attacking Large Language Models with Projected Gradient Descent**

使用投影梯度下降攻击大型语言模型 cs.LG

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2402.09154v2) [paper-pdf](http://arxiv.org/pdf/2402.09154v2)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.

摘要: 当前的LLM对齐方法很容易通过专门设计的对抗提示来突破。虽然使用离散优化制作对抗提示非常有效，但此类攻击通常使用超过100，000次LLM调用。这种高计算成本使它们不适合例如定量分析和对抗训练。为了解决这个问题，我们在持续放松的输入提示下重新审视投影梯度下降（PVD）。尽管之前对普通的基于梯度的攻击的尝试基本上失败了，但我们表明，仔细控制持续放松带来的错误可以极大地提高它们的功效。我们的LLM PGO比最先进的离散优化快一个数量级，以实现相同的毁灭性攻击结果。



## **10. PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs**

PAPILLON：针对LLM的高效、隐蔽的Fuzz测试动力越狱 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.14866v5) [paper-pdf](http://arxiv.org/pdf/2409.14866v5)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs. In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses. Code: https://github.com/aaFrostnova/Papillon

摘要: 大型语言模型(LLM)在各种任务中表现出色，但仍然容易受到越狱攻击，在越狱攻击中，攻击者创建越狱提示来误导模型生成有害或攻击性内容。当前的越狱方法要么严重依赖于人工制作的模板，这对可伸缩性和适应性构成了挑战，要么难以生成语义连贯的提示，使它们很容易被检测到。此外，大多数现有方法都需要冗长的提示，从而导致更高的查询成本。为了应对这些挑战，我们引入了一种新的越狱攻击框架Papillon，它是一个自动化的黑盒越狱攻击框架，采用了一系列定制的设计来适应黑盒模糊测试方法。与依赖手工制作的模板不同，Papillon从一个空的种子库开始，不需要搜索任何相关的越狱模板。我们还开发了三种新的问题相关突变策略，使用LLM助手来生成提示，这些提示在保持语义连贯的同时显著缩短了提示的长度。此外，我们实现了一个两级判断模块来准确地检测真正的成功越狱。我们在7个有代表性的LLM上对Papillon进行了评估，并将其与5种最先进的越狱攻击策略进行了比较。对于专有的LLMAPI，如GPT-3.5 Turbo、GPT-4和Gemini-Pro，Papillons的攻击成功率分别超过90%、80%和74%，比现有基线高出60%以上。此外，Papillon可以保持高度的语义连贯性，同时显著缩短越狱提示的长度。当针对GPT-4时，Papillon即使使用100个令牌也可以达到78%以上的攻击成功率。此外，乳突展示了可转移性，并对最先进的防御措施具有很强的抵抗力。代码：https://github.com/aaFrostnova/Papillon



## **11. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

对LLM as-a-Judge的基于优化的即时注入攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2403.17710v4) [paper-pdf](http://arxiv.org/pdf/2403.17710v4)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.

摘要: LLM-as-a-Court使用大型语言模型(LLM)从给定问题的一组候选人中选择最佳答案。LLM-as-a-Court有许多应用，如LLM支持的搜索、带人工智能反馈的强化学习(RLAIF)和工具选择。在这项工作中，我们提出了一种针对LLM-as-a-Court的基于优化的快速注入攻击--JudgeDeceiver。JudgeDeceiver将精心制作的序列注入到攻击者控制的候选响应中，以便LLM-as-a-Court为攻击者选择的问题选择候选响应，而不管其他候选响应是什么。具体地说，我们将寻找这样的序列描述为一个优化问题，并提出了一种基于梯度的方法来近似求解它。我们的广泛评估表明，JudgeDecept是非常有效的，并且比现有的手动手工创建注入序列的即时注入攻击和越狱攻击更有效，当扩展到我们的问题时。我们还在三个案例研究中展示了JudgeDeceiver的有效性，即LLM支持的搜索、RLAIF和工具选择。此外，我们还考虑了防御措施，包括已知答案检测、困惑检测和困惑加窗检测。我们的结果表明，这些防御措施是不够的，这突显了开发新的防御战略的迫切需要。我们的实现可从以下存储库获得：https://github.com/ShiJiawenwen/JudgeDeceiver.



## **12. Exploring Adversarial Robustness in Classification tasks using DNA Language Models**

使用DNA语言模型探索分类任务中的对抗鲁棒性 cs.CL

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2409.19788v2) [paper-pdf](http://arxiv.org/pdf/2409.19788v2)

**Authors**: Hyunwoo Yoo, Haebin Shin, Kaidi Xu, Gail Rosen

**Abstract**: DNA Language Models, such as GROVER, DNABERT2 and the Nucleotide Transformer, operate on DNA sequences that inherently contain sequencing errors, mutations, and laboratory-induced noise, which may significantly impact model performance. Despite the importance of this issue, the robustness of DNA language models remains largely underexplored. In this paper, we comprehensivly investigate their robustness in DNA classification by applying various adversarial attack strategies: the character (nucleotide substitutions), word (codon modifications), and sentence levels (back-translation-based transformations) to systematically analyze model vulnerabilities. Our results demonstrate that DNA language models are highly susceptible to adversarial attacks, leading to significant performance degradation. Furthermore, we explore adversarial training method as a defense mechanism, which enhances both robustness and classification accuracy. This study highlights the limitations of DNA language models and underscores the necessity of robustness in bioinformatics.

摘要: DNA语言模型，如Grover、DNABERT2和核苷酸转换器，对DNA序列进行操作，这些序列固有地包含测序错误、突变和实验室诱导的噪声，这些可能会显著影响模型的性能。尽管这个问题很重要，但DNA语言模型的稳健性在很大程度上仍然没有得到充分的研究。在本文中，我们通过应用各种对抗性攻击策略：字符(核苷酸替换)、单词(密码子修改)和句子级别(基于反向翻译的转换)来系统地分析模型的脆弱性，全面地研究了它们在DNA分类中的稳健性。我们的结果表明，DNA语言模型非常容易受到对抗性攻击，导致性能显著下降。此外，我们还探索了对抗性训练方法作为一种防御机制，提高了鲁棒性和分类准确率。这项研究突出了DNA语言模型的局限性，并强调了生物信息学中稳健性的必要性。



## **13. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

MAA：针对视觉语言预训练模型的强力对抗攻击 cs.CV

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2502.08079v3) [paper-pdf](http://arxiv.org/pdf/2502.08079v3)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.

摘要: 当前用于评估视觉语言预训练(VLP)模型在多模式任务中的稳健性的对抗性攻击存在可转移性有限的问题，其中针对特定模型的攻击往往难以在不同的模型上有效地泛化，从而限制了它们在更广泛地评估稳健性方面的有效性。这主要归因于过度依赖特定型号的特征和区域，特别是在图像模式方面。在本文中，我们提出了一种优雅而高效的方法，称为精细攻击(MAA)，它充分利用了个体样本的模型无关特性和脆弱性，从而增强了泛化能力，降低了模型依赖。MAA通过开发一种新的调整大小和滑动裁剪(RSCrop)技术，结合多粒度相似破坏(MGSD)策略，强调对抗性图像的细粒度优化。在不同的VLP模型、多个基准数据集和各种下游任务上的广泛实验表明，MAA显著增强了对抗性攻击的有效性和可转移性。我们进行了大量的性能研究，以深入了解各种型号配置的有效性，从而指导该领域的未来发展。



## **14. We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs**

我们为您准备了一个套餐！通过代码生成LLM综合分析包幻觉 cs.SE

To appear in the 2025 USENIX Security Symposium. 22 pages, 14  figures, 8 tables. Edited from original version for submission to a different  conference. No change to original results or findings

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2406.10279v3) [paper-pdf](http://arxiv.org/pdf/2406.10279v3)

**Authors**: Joseph Spracklen, Raveen Wijewickrama, A H M Nazmus Sakib, Anindya Maiti, Bimal Viswanath, Murtuza Jadliwala

**Abstract**: The reliance of popular programming languages such as Python and JavaScript on centralized package repositories and open-source software, combined with the emergence of code-generating Large Language Models (LLMs), has created a new type of threat to the software supply chain: package hallucinations. These hallucinations, which arise from fact-conflicting errors when generating code using LLMs, represent a novel form of package confusion attack that poses a critical threat to the integrity of the software supply chain. This paper conducts a rigorous and comprehensive evaluation of package hallucinations across different programming languages, settings, and parameters, exploring how a diverse set of models and configurations affect the likelihood of generating erroneous package recommendations and identifying the root causes of this phenomenon. Using 16 popular LLMs for code generation and two unique prompt datasets, we generate 576,000 code samples in two programming languages that we analyze for package hallucinations. Our findings reveal that that the average percentage of hallucinated packages is at least 5.2% for commercial models and 21.7% for open-source models, including a staggering 205,474 unique examples of hallucinated package names, further underscoring the severity and pervasiveness of this threat. To overcome this problem, we implement several hallucination mitigation strategies and show that they are able to significantly reduce the number of package hallucinations while maintaining code quality. Our experiments and findings highlight package hallucinations as a persistent and systemic phenomenon while using state-of-the-art LLMs for code generation, and a significant challenge which deserves the research community's urgent attention.

摘要: 流行的编程语言，如Python和JavaScript对集中包库和开源软件的依赖，再加上代码生成大型语言模型(LLM)的出现，对软件供应链造成了一种新的威胁：包幻觉。这些幻觉是由使用LLMS生成代码时与事实冲突的错误引起的，代表了一种新形式的包混淆攻击，对软件供应链的完整性构成了严重威胁。本文对不同编程语言、设置和参数的套餐幻觉进行了严格和全面的评估，探索了不同的模型和配置如何影响生成错误套餐推荐的可能性，并找出了这种现象的根本原因。使用16个流行的LLM进行代码生成和两个独特的提示数据集，我们用两种编程语言生成了576,000个代码样本，并分析了程序包幻觉。我们的调查结果显示，商业型号的幻觉包平均比例至少为5.2%，开源型号的平均幻觉包比例为21.7%，其中包括惊人的205,474个幻觉包名称的独特例子，进一步突显了这一威胁的严重性和普遍性。为了克服这个问题，我们实现了几种幻觉缓解策略，并表明它们能够在保持代码质量的同时显著减少包幻觉的数量。我们的实验和发现突出了程序包幻觉是一种持续和系统的现象，同时使用最先进的LLM来生成代码，这是一个值得研究界紧急关注的重大挑战。



## **15. Boosting Jailbreak Attack with Momentum**

以势头助推越狱攻击 cs.LG

Accepted by ICASSP 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2405.01229v2) [paper-pdf](http://arxiv.org/pdf/2405.01229v2)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-known jailbreak attack. In particular, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of the adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous optimization iterations. Specifically, we propose the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which integrates a momentum term into the gradient heuristic to boost and stabilize the random search for tokens in adversarial prompts. Experimental results showcase the notable enhancement achieved by MAC over baselines in terms of attack success rate and optimization efficiency. Moreover, we demonstrate that MAC can still exhibit superior performance for transfer attacks and models under defense mechanisms. Our code is available at https://github.com/weizeming/momentum-attack-llm.

摘要: 大型语言模型(LLM)在不同的任务中取得了显著的成功，但它们仍然容易受到对手攻击，特别是众所周知的越狱攻击。特别是，贪婪坐标梯度(GCG)攻击已经证明了通过结合梯度启发式和贪婪搜索来优化敌意提示来利用该漏洞的有效性。然而，这种攻击的效率已经成为攻击过程中的瓶颈。为了缓解这一局限性，在本文中，我们通过优化镜头重新考虑敌意提示的生成，目的是稳定优化过程，并从先前的优化迭代中获得更多启发式的见解。具体地说，我们提出了加速G/Textbf{C}G(Textbf{MAC})攻击，该攻击将动量项融入到梯度启发式中，以增强和稳定敌意提示中随机搜索令牌的能力。实验结果表明，在攻击成功率和优化效率方面，MAC算法在攻击成功率和优化效率方面都有明显的提高。此外，我们还证明了在防御机制下，MAC对于传输攻击和模型仍然表现出优越的性能。我们的代码可以在https://github.com/weizeming/momentum-attack-llm.上找到



## **16. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

CLIPure：通过CLIP在潜空间中净化，以实现对抗鲁棒零镜头分类 cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2502.18176v2) [paper-pdf](http://arxiv.org/pdf/2502.18176v2)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.

摘要: 在这篇文章中，我们的目标是建立一个对抗性稳健的零镜头图像分类器。我们的工作基于CLIP，这是一个视觉语言预先训练的编码器模型，它可以通过将图像与文本提示进行匹配来执行零镜头分类。净化是我们选择的路径，因为它不需要针对特定攻击类型的对抗性训练，因此可以应对任何可预见的攻击。然后，我们通过双向随机微分方程(SDE)将净化风险表示为对敌方样本去噪的净化过程和对良性样本添加扰动的攻击过程的联合分布之间的KL发散。最终得出的结果启发我们去探索CLIP的多峰潜伏空间中的净化。我们为我们的CLIPure方法提出了两种变体：CLIPure-Diff和CLIPure-Cos，CLIPure-Diff使用DALE-2中的DiffusionPrior模块(对剪辑的潜在向量的生成过程进行建模)来模拟图像的潜在向量的可能性，CLIPure-Cos使用图像的嵌入和“a的照片”之间的余弦相似性来建模可能性。据我们所知，CLIPure是第一个在多峰潜在空间中进行净化的方法，而CLIPure-Cos是第一个不基于产生式模型的净化方法，大大提高了防御效率。我们在CIFAR-10、ImageNet和13个数据集上进行了广泛的实验，这些数据集是以前基于剪辑的防御方法用于评估零镜头分类稳健性的。结果表明，CLIPure在很大程度上提高了SOTA的健壮性，例如，在CIFAR10上从71.7%提高到91.1%，在ImageNet上从59.6%提高到72.6%，在13个数据集上的平均健壮性比以前的SOTA提高了108%。代码可在https://github.com/TMLResearchGroup-CAS/CLIPure.上获得



## **17. Output Length Effect on DeepSeek-R1's Safety in Forced Thinking**

输出长度对DeepSeek-R1在强迫思维中安全性的影响 cs.CL

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.01923v1) [paper-pdf](http://arxiv.org/pdf/2503.01923v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Victor Bian

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities, but their safety under adversarial conditions remains a challenge. This study examines the impact of output length on the robustness of DeepSeek-R1, particularly in Forced Thinking scenarios. We analyze responses across various adversarial prompts and find that while longer outputs can improve safety through self-correction, certain attack types exploit extended generations. Our findings suggest that output length should be dynamically controlled to balance reasoning effectiveness and security. We propose reinforcement learning-based policy adjustments and adaptive token length regulation to enhance LLM safety.

摘要: 大型语言模型（LLM）已表现出强大的推理能力，但它们在对抗条件下的安全性仍然是一个挑战。本研究考察了输出长度对DeepSeek-R1稳健性的影响，特别是在强迫思维场景中。我们分析了各种对抗提示的响应，发现虽然更长的输出可以通过自我纠正来提高安全性，但某些攻击类型会利用延长的世代。我们的研究结果表明，应该动态控制输出长度，以平衡推理有效性和安全性。我们提出基于强化学习的政策调整和自适应代币长度监管，以增强LLM安全性。



## **18. SeqAR: Jailbreak LLMs with Sequential Auto-Generated Characters**

SeqAR：具有连续自动生成角色的越狱LLMS cs.CR

Accepted by NAACL 2025

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2407.01902v2) [paper-pdf](http://arxiv.org/pdf/2407.01902v2)

**Authors**: Yan Yang, Zeguan Xiao, Xin Lu, Hongru Wang, Xuetao Wei, Hailiang Huang, Guanhua Chen, Yun Chen

**Abstract**: The widespread applications of large language models (LLMs) have brought about concerns regarding their potential misuse. Although aligned with human preference data before release, LLMs remain vulnerable to various malicious attacks. In this paper, we adopt a red-teaming strategy to enhance LLM safety and introduce SeqAR, a simple yet effective framework to design jailbreak prompts automatically. The SeqAR framework generates and optimizes multiple jailbreak characters and then applies sequential jailbreak characters in a single query to bypass the guardrails of the target LLM. Different from previous work which relies on proprietary LLMs or seed jailbreak templates crafted by human expertise, SeqAR can generate and optimize the jailbreak prompt in a cold-start scenario using open-sourced LLMs without any seed jailbreak templates. Experimental results show that SeqAR achieves attack success rates of 88% and 60% in bypassing the safety alignment of GPT-3.5-1106 and GPT-4, respectively. Furthermore, we extensively evaluate the transferability of the generated templates across different LLMs and held-out malicious requests, while also exploring defense strategies against the jailbreak attack designed by SeqAR.

摘要: 大型语言模型(LLM)的广泛应用引起了人们对其潜在滥用的担忧。尽管在发布之前与人类偏好数据保持一致，但LLM仍然容易受到各种恶意攻击。在本文中，我们采用了红队策略来增强LLM的安全性，并引入了一个简单而有效的框架SeqAR来自动设计越狱提示。SeqAR框架生成并优化多个越狱字符，然后在单个查询中应用连续的越狱字符以绕过目标LLM的护栏。不同于以往依赖专有LLM或人工制作的种子越狱模板的工作，SeqAR可以在冷启动场景下使用开源LLMS生成和优化越狱提示，而不需要任何种子越狱模板。实验结果表明，SeqAR在绕过GPT-3.5-1106和GPT-4安全对齐的攻击成功率分别达到88%和60%。此外，我们还广泛评估了生成的模板在不同LLM和拒绝恶意请求之间的可转移性，同时也探索了针对SeqAR设计的越狱攻击的防御策略。



## **19. Unmasking Digital Falsehoods: A Comparative Analysis of LLM-Based Misinformation Detection Strategies**

揭露数字谎言：基于LLM的错误信息检测策略的比较分析 cs.CL

**SubmitDate**: 2025-03-02    [abs](http://arxiv.org/abs/2503.00724v1) [paper-pdf](http://arxiv.org/pdf/2503.00724v1)

**Authors**: Tianyi Huang, Jingyuan Yi, Peiyang Yu, Xiaochuan Xu

**Abstract**: The proliferation of misinformation on social media has raised significant societal concerns, necessitating robust detection mechanisms. Large Language Models such as GPT-4 and LLaMA2 have been envisioned as possible tools for detecting misinformation based on their advanced natural language understanding and reasoning capabilities. This paper conducts a comparison of LLM-based approaches to detecting misinformation between text-based, multimodal, and agentic approaches. We evaluate the effectiveness of fine-tuned models, zero-shot learning, and systematic fact-checking mechanisms in detecting misinformation across different topic domains like public health, politics, and finance. We also discuss scalability, generalizability, and explainability of the models and recognize key challenges such as hallucination, adversarial attacks on misinformation, and computational resources. Our findings point towards the importance of hybrid approaches that pair structured verification protocols with adaptive learning techniques to enhance detection accuracy and explainability. The paper closes by suggesting potential avenues of future work, including real-time tracking of misinformation, federated learning, and cross-platform detection models.

摘要: 社交媒体上虚假信息的泛滥引发了重大的社会担忧，需要强有力的检测机制。大型语言模型，如GPT-4和LLaMA2，已被设想为基于其先进的自然语言理解和推理能力而可能用于检测错误信息的工具。本文对基于LLM的错误信息检测方法在基于文本的方法、多通道方法和代理方法之间进行了比较。我们评估了微调模型、零距离学习和系统的事实核查机制在检测公共卫生、政治和金融等不同主题领域的错误信息方面的有效性。我们还讨论了模型的可伸缩性、通用性和可解释性，并认识到关键挑战，如幻觉、对错误信息的敌意攻击和计算资源。我们的发现指出了将结构化验证协议与自适应学习技术配对以提高检测准确性和可解释性的混合方法的重要性。论文最后提出了未来工作的潜在途径，包括错误信息的实时跟踪、联邦学习和跨平台检测模型。



## **20. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

这是谁写的？零镜头LLM生成文本检测的关键是GECScore cs.CL

COLING 2025

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2405.04286v2) [paper-pdf](http://arxiv.org/pdf/2405.04286v2)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of detectors for texts generated by large language models (LLMs) substantially depends on the availability of large-scale training data. However, white-box zero-shot detectors, which require no such data, are limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose a simple yet effective black-box zero-shot detection approach based on the observation that, from the perspective of LLMs, human-written texts typically contain more grammatical errors than LLM-generated texts. This approach involves calculating the Grammar Error Correction Score (GECScore) for the given text to differentiate between human-written and LLM-generated text. Experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.62% across XSum and Writing Prompts dataset. Additionally, our approach demonstrates strong reliability in the wild, exhibiting robust generalization and resistance to paraphrasing attacks. Data and code are available at: https://github.com/NLP2CT/GECScore.

摘要: 对大型语言模型(LLM)生成的文本的检测器的有效性在很大程度上取决于大规模训练数据的可用性。然而，白盒零激发探测器不需要这样的数据，受到LLM生成的文本的源模型的可访问性的限制。在本文中，我们提出了一种简单而有效的黑盒零镜头检测方法，从LLMS的角度来看，人类书写的文本通常比LLM生成的文本包含更多的语法错误。这种方法包括计算给定文本的语法纠错分数(GECScore)，以区分人类编写的文本和LLM生成的文本。实验结果表明，该方法在XSum和Writing Prompt数据集上的平均AUROC达到了98.62%，优于目前最先进的SOTA零镜头和监督方法。此外，我们的方法在野外表现出很强的可靠性，表现出健壮的泛化和对意译攻击的抵抗。有关数据和代码，请访问：https://github.com/NLP2CT/GECScore.。



## **21. Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark**

修改和生成文本检测：通过水印实现LLM输出的双重检测能力 cs.CR

**SubmitDate**: 2025-03-01    [abs](http://arxiv.org/abs/2502.08332v2) [paper-pdf](http://arxiv.org/pdf/2502.08332v2)

**Authors**: Yuhang Cai, Yaofei Wang, Donghui Hu, Chen Gu

**Abstract**: The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.

摘要: 大型语言模型(LLM)的发展引起了人们对潜在滥用的担忧。一种实用的解决方案是在文本中嵌入水印，允许通过提取水印来验证所有权。现有的方法主要集中在防御修改攻击上，往往忽略了其他欺骗攻击。例如，攻击者可以更改带水印的文本以产生有害内容，而不会影响水印的存在，这可能会导致将此恶意内容错误地归因于LLM。这种情况对LLMS服务提供商构成了严重威胁，并突出了同时实现修改检测和生成文本检测的重要性。因此，我们提出了一种文本修改检测技术，以检测对修改敏感的无偏水印。提出了一种新的水印检测方法--“丢弃令牌”，该度量度量了水印检测中未包含的令牌个数。当水印发生修改时，该度量会发生变化，并且可以作为修改的证据。此外，我们对水印检测过程进行了改进，提出了一种新的无偏水印检测方法。实验表明，我们可以实现有效的双重检测能力：修改检测和水印生成文本检测。



## **22. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

保护人工智能代理：开发和分析安全架构 cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2409.03793v3) [paper-pdf](http://arxiv.org/pdf/2409.03793v3)

**Authors**: Ishaan Domkundwar, Mukunda N S, Ishaan Bhola, Riddhik Kochhar

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.

摘要: 人工智能代理，特别是由大型语言模型驱动的，在需要精确度和效率的各种应用中展示了非凡的能力。然而，这些代理伴随着固有的风险，包括潜在的不安全或有偏见的行动，易受对手攻击，缺乏透明度，以及产生幻觉的倾向。随着人工智能代理在该行业的关键部门变得越来越普遍，实施有效的安全协议变得越来越重要。本文讨论了人工智能系统中安全措施的迫切需要，特别是与人类团队协作的系统。我们提出并评估了三个框架来增强AI代理系统中的安全协议：LLM驱动的输入输出过滤器、集成在系统中的安全代理以及嵌入安全检查的基于分级委托的系统。我们的方法涉及实现这些框架并针对一组不安全的代理用例对它们进行测试，提供对它们在降低与AI代理部署相关的风险方面的有效性的全面评估。我们的结论是，这些框架可以显著加强AI代理系统的安全性和安全性，将潜在的有害行为或输出降至最低。我们的工作有助于持续努力创建安全可靠的人工智能应用程序，特别是在自动化操作中，并为开发强大的护栏提供基础，以确保在现实世界的应用程序中负责任地使用人工智能代理。



## **23. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning**

UPora：通过动态劫持LLM代理自己的推理来对抗他们的统一红色团队框架 cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2503.01908v1) [paper-pdf](http://arxiv.org/pdf/2503.01908v1)

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for handling complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements also amplify the risks of adversarial attacks, particularly when LLM agents can access sensitive external functionalities. Moreover, because LLM agents engage in extensive reasoning or planning before executing final actions, manipulating them into performing targeted malicious actions or invoking specific tools remains a significant challenge. Consequently, directly embedding adversarial strings in malicious instructions or injecting malicious prompts into tool interactions has become less effective against modern LLM agents. In this work, we present UDora, a unified red teaming framework designed for LLM Agents that dynamically leverages the agent's own reasoning processes to compel it toward malicious behavior. Specifically, UDora first samples the model's reasoning for the given task, then automatically identifies multiple optimal positions within these reasoning traces to insert targeted perturbations. Subsequently, it uses the modified reasoning as the objective to optimize the adversarial strings. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets.

摘要: 配备了外部工具的大型语言模型(LLM)代理在处理网络购物、自动回复电子邮件和金融交易等复杂任务方面变得越来越强大。然而，这些进步也放大了对抗性攻击的风险，特别是当LLM特工可以访问敏感的外部功能时。此外，由于LLM代理在执行最终操作之前会进行广泛的推理或规划，因此操纵它们执行有针对性的恶意操作或调用特定工具仍然是一个重大挑战。因此，直接在恶意指令中嵌入敌意字符串或在工具交互中插入恶意提示已变得对现代LLM代理不那么有效。在这项工作中，我们提出了Udora，一个为LLM代理设计的统一的红色团队框架，它动态地利用代理自己的推理过程来迫使其走向恶意行为。具体地说，Udora首先对给定任务的模型推理进行采样，然后自动在这些推理轨迹中识别多个最佳位置，以插入有针对性的扰动。随后，以改进后的推理为目标，对对抗性字符串进行优化。通过反复应用此过程，LLM代理随后将被诱导执行指定的恶意操作或调用特定的恶意工具。与现有方法相比，我们的方法在三个LLM试剂数据集上表现出了更好的有效性。



## **24. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

引导对话动力学，增强抵御多回合越狱攻击的稳健性 cs.CL

28 pages, 10 figures, 7 tables

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2503.00187v1) [paper-pdf](http://arxiv.org/pdf/2503.00187v1)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are highly vulnerable to jailbreaking attacks, wherein adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off between safety and helpfulness under different multi-turn jailbreak methods. Our code is available at https://github.com/HanjiangHu/NBF-LLM .

摘要: 大型语言模型(LLM)非常容易受到越狱攻击，在越狱攻击中，敌意提示旨在引发有害的响应。虽然现有的防御系统通过检测和过滤不安全的输入有效地缓解了单回合攻击，但它们无法抵御利用多个交互中的上下文漂移的多回合越狱，逐渐导致LLM远离安全行为。为了应对这一挑战，我们提出了一种基于安全控制理论的安全转向框架，以确保多轮对话中的恒定安全。我们的方法使用状态空间表示来模拟与LLMS的对话，并引入了一种新的神经屏障函数(NBF)来主动检测和过滤从不断变化的上下文中出现的有害查询。我们的方法通过学习解释敌意查询的安全预测器，防止潜在的上下文漂移到越狱，在每一轮对话中实现不变的安全。在多个LLM下的广泛实验表明，我们的基于NBF的安全转向性能优于安全对齐基线，提供了更强大的防御多转弯越狱的同时，在不同的多转弯越狱方法下保持了安全和帮助之间的更好权衡。我们的代码可以在https://github.com/HanjiangHu/NBF-LLM上找到。



## **25. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

Logicbreaks：理解基于规则的推理颠覆的框架 cs.AI

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2407.00075v5) [paper-pdf](http://arxiv.org/pdf/2407.00075v5)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.

摘要: 我们研究如何根据预算指定的规则颠覆大型语言模型（LLM）。我们首先将规则遵循形式化为命题Horn逻辑中的推理，这是一个数学系统，其中规则的形式为“如果$P$和$Q$，那么$R$”，对于某些命题$P$、$Q$和$R$。接下来，我们证明，尽管小型变压器可以忠实地遵循这些规则，但恶意制作的提示仍然会误导理论构建和从数据中学习的模型。此外，我们证明了LLM上的流行攻击算法可以找到对抗提示并诱导与我们的理论一致的注意力模式。我们新颖的基于逻辑的框架为在基于规则的环境中研究LLM提供了基础，从而能够对逻辑推理和越狱攻击等任务进行正式分析。



## **26. Learning diverse attacks on large language models for robust red-teaming and safety tuning**

学习对大型语言模型的多样化攻击，以实现强大的红色团队化和安全调整 cs.CL

ICLR 2025

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2405.18540v2) [paper-pdf](http://arxiv.org/pdf/2405.18540v2)

**Authors**: Seanie Lee, Minsu Kim, Lynn Cherif, David Dobre, Juho Lee, Sung Ju Hwang, Kenji Kawaguchi, Gauthier Gidel, Yoshua Bengio, Nikolay Malkin, Moksh Jain

**Abstract**: Red-teaming, or identifying prompts that elicit harmful responses, is a critical step in ensuring the safe and responsible deployment of large language models (LLMs). Developing effective protection against many modes of attack prompts requires discovering diverse attacks. Automated red-teaming typically uses reinforcement learning to fine-tune an attacker language model to generate prompts that elicit undesirable responses from a target LLM, as measured, for example, by an auxiliary toxicity classifier. We show that even with explicit regularization to favor novelty and diversity, existing approaches suffer from mode collapse or fail to generate effective attacks. As a flexible and probabilistically principled alternative, we propose to use GFlowNet fine-tuning, followed by a secondary smoothing phase, to train the attacker model to generate diverse and effective attack prompts. We find that the attacks generated by our method are effective against a wide range of target LLMs, both with and without safety tuning, and transfer well between target LLMs. Finally, we demonstrate that models safety-tuned using a dataset of red-teaming prompts generated by our method are robust to attacks from other RL-based red-teaming approaches.

摘要: 红色团队，或识别引发有害响应的提示，是确保安全和负责任地部署大型语言模型(LLM)的关键步骤。开发针对多种攻击提示的有效防护需要发现不同的攻击。自动红色团队通常使用强化学习来微调攻击者语言模型，以生成引发来自目标LLM的不良响应的提示，例如通过辅助毒性分类器来测量。我们表明，即使使用显式正则化来支持新颖性和多样性，现有的方法也会遭受模式崩溃或无法产生有效的攻击。作为一种灵活的、符合概率原则的替代方案，我们建议使用GFlowNet微调，然后进行二次平滑阶段，来训练攻击者模型以生成多样化和有效的攻击提示。我们发现，我们的方法产生的攻击对大范围的目标LLM有效，无论是否进行安全调整，并在目标LLM之间很好地转移。最后，我们证明了使用我们的方法生成的红队提示的数据集进行安全调整的模型对于来自其他基于RL的红队方法的攻击是健壮的。



## **27. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

LLM Whisperer：对LLM偏见回应的不起眼攻击 cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2406.04755v4) [paper-pdf](http://arxiv.org/pdf/2406.04755v4)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing that our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.

摘要: 为大型语言模型(LLM)编写有效的提示可能是不直观和繁琐的。作为回应，优化或建议提示的服务应运而生。虽然这类服务可以减少用户的工作，但它们也带来了风险：提示提供商可能会巧妙地操纵提示，以产生严重偏见的LLM响应。在这项工作中，我们表明，提示中微妙的同义词替换可以增加LLMS提到目标概念(例如，品牌、政党、国家)的可能性(差异高达78%)。我们通过一项用户研究证实了我们的观察结果，表明我们被敌意干扰的提示1)与人类未改变的提示难以区分，2)推动LLM更频繁地推荐目标概念，3)使用户更有可能注意到目标概念，所有这些都不会引起怀疑。这种攻击的实用性有可能破坏用户的自主性。在其他措施中，我们建议实施警告，以防止使用来自不受信任方的提示。



## **28. FC-Attack: Jailbreaking Large Vision-Language Models via Auto-Generated Flowcharts**

FC攻击：通过自动生成流程图破解大型视觉语言模型 cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.21059v1) [paper-pdf](http://arxiv.org/pdf/2502.21059v1)

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Jihui Guo, Xinlei He

**Abstract**: Large Vision-Language Models (LVLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most LVLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, LVLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute a jailbreak attack on LVLMs. Our evaluations using the Advbench dataset show that FC-Attack achieves over 90% attack success rates on Gemini-1.5, Llaval-Next, Qwen2-VL, and InternVL-2.5 models, outperforming existing LVLM jailbreak methods. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. Our evaluation shows that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop.

摘要: 大型视觉语言模型在一些实际应用中已经变得功能强大并被广泛采用。然而，最近的研究揭示了它们在多模式越狱攻击中的脆弱性，即该模型可能被诱导生成有害内容，从而导致安全风险。尽管大多数LVLM都经过了安全调整，但最近的研究表明，视觉通道仍然容易受到越狱攻击。在我们的工作中，我们发现，通过使用带有部分有害信息的流程图，可以诱导LVLM提供额外的有害细节。在此基础上，提出了一种基于自动生成流程图的越狱攻击方法FC-Attack。具体地说，FC-Attack首先微调预先训练的LLM，以创建基于良性数据集的步骤描述生成器。然后，生成器被用来生成与有害查询相对应的步骤描述，这些描述被转换为3种不同形状(垂直、水平和S形状)的流程图作为视觉提示。然后，将这些流程图与良性文本提示相结合，以执行对LVLMS的越狱攻击。我们使用Advbench数据集进行的评估表明，FC-Attack在Gemini-1.5、Llaval-Next、Qwen2-VL和InternVL-2.5模型上的攻击成功率超过90%，性能优于现有的LVLM越狱方法。此外，我们还研究了影响攻击性能的因素，包括步骤数和流程图中的字体样式。我们的评估表明，FC-Attack可以通过改变字体风格将Claude-3.5的越狱性能从4%提高到28%。为了缓解攻击，我们探索了几种防御措施，发现AdaShield可以在很大程度上降低越狱性能，但代价是效用下降。



## **29. Behind the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models**

效率提示背后：揭露小型语言模型中越狱攻击的潜在威胁 cs.CR

12 pages. 6 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.19883v2) [paper-pdf](http://arxiv.org/pdf/2502.19883v2)

**Authors**: Sibo Yi, Tianshuo Cong, Xinlei He, Qi Li, Jiaxing Song

**Abstract**: Small language models (SLMs) have become increasingly prominent in the deployment on edge devices due to their high efficiency and low computational cost. While researchers continue to advance the capabilities of SLMs through innovative training strategies and model compression techniques, the security risks of SLMs have received considerably less attention compared to large language models (LLMs).To fill this gap, we provide a comprehensive empirical study to evaluate the security performance of 13 state-of-the-art SLMs under various jailbreak attacks. Our experiments demonstrate that most SLMs are quite susceptible to existing jailbreak attacks, while some of them are even vulnerable to direct harmful prompts.To address the safety concerns, we evaluate several representative defense methods and demonstrate their effectiveness in enhancing the security of SLMs. We further analyze the potential security degradation caused by different SLM techniques including architecture compression, quantization, knowledge distillation, and so on. We expect that our research can highlight the security challenges of SLMs and provide valuable insights to future work in developing more robust and secure SLMs.

摘要: 小语言模型(SLM)以其高效率和低计算成本在边缘设备上的部署中日益突出。虽然研究人员不断通过创新的训练策略和模型压缩技术来提升SLM的能力，但与大型语言模型(LLM)相比，SLM的安全风险受到的关注要少得多。为了填补这一空白，我们提供了一项全面的实证研究，评估了13种最先进的SLM在各种越狱攻击下的安全性能。我们的实验表明，大多数SLM很容易受到现有越狱攻击，其中一些甚至容易受到直接有害提示的攻击。为了解决安全问题，我们评估了几种有代表性的防御方法，并证明了它们在增强SLM安全性方面的有效性。我们进一步分析了不同的SLM技术，包括体系结构压缩、量化、知识提取等所造成的潜在的安全降级。我们期望我们的研究能够突出SLM的安全挑战，并为未来开发更强大和安全的SLM的工作提供有价值的见解。



## **30. LoRec: Large Language Model for Robust Sequential Recommendation against Poisoning Attacks**

LoRec：针对中毒攻击的鲁棒顺序推荐的大型语言模型 cs.IR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2401.17723v2) [paper-pdf](http://arxiv.org/pdf/2401.17723v2)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Sequential recommender systems stand out for their ability to capture users' dynamic interests and the patterns of item-to-item transitions. However, the inherent openness of sequential recommender systems renders them vulnerable to poisoning attacks, where fraudulent users are injected into the training data to manipulate learned patterns. Traditional defense strategies predominantly depend on predefined assumptions or rules extracted from specific known attacks, limiting their generalizability to unknown attack types. To solve the above problems, considering the rich open-world knowledge encapsulated in Large Language Models (LLMs), our research initially focuses on the capabilities of LLMs in the detection of unknown fraudulent activities within recommender systems, a strategy we denote as LLM4Dec. Empirical evaluations demonstrate the substantial capability of LLMs in identifying unknown fraudsters, leveraging their expansive, open-world knowledge.   Building upon this, we propose the integration of LLMs into defense strategies to extend their effectiveness beyond the confines of known attacks. We propose LoRec, an advanced framework that employs LLM-Enhanced Calibration to strengthen the robustness of sequential recommender systems against poisoning attacks. LoRec integrates an LLM-enhanced CalibraTor (LCT) that refines the training process of sequential recommender systems with knowledge derived from LLMs, applying a user-wise reweighting to diminish the impact of fraudsters injected by attacks. By incorporating LLMs' open-world knowledge, the LCT effectively converts the limited, specific priors or rules into a more general pattern of fraudsters, offering improved defenses against poisoning attacks. Our comprehensive experiments validate that LoRec, as a general framework, significantly strengthens the robustness of sequential recommender systems.

摘要: 顺序推荐系统因其能够捕获用户的动态兴趣和项到项转换的模式而脱颖而出。然而，顺序推荐系统固有的开放性使得它们容易受到中毒攻击，在这种攻击中，欺诈性用户被注入到训练数据中以操纵学习模式。传统的防御策略主要依赖于从特定已知攻击中提取的预定义假设或规则，将其泛化为未知攻击类型。为了解决上述问题，考虑到大型语言模型(LLMS)中封装的丰富的开放世界知识，我们的研究最初集中在LLMS对推荐系统中未知欺诈活动的检测能力，我们将其命名为LLM4Dec。经验评估表明，LLMS利用其广博的、开放的知识，在识别未知欺诈者方面具有很强的能力。在此基础上，我们建议将LLM整合到防御战略中，以将其有效性扩展到已知攻击的范围之外。我们提出了LoRec，这是一个先进的框架，它使用LLM增强的校准来增强序列推荐系统对中毒攻击的健壮性。LoRec集成了LLM增强的校准器(LCT)，该校准器利用来自LLMS的知识来优化顺序推荐系统的训练过程，应用用户级的重新加权来减少攻击注入的欺诈者的影响。通过融入LLMS的开放世界知识，LCT有效地将有限的、特定的先例或规则转换为更一般的欺诈者模式，提供更好的防御中毒攻击的能力。我们的综合实验证明，LoRec作为一个通用的框架，显著增强了序列推荐系统的健壮性。



## **31. Efficient Jailbreaking of Large Models by Freeze Training: Lower Layers Exhibit Greater Sensitivity to Harmful Content**

通过冷冻训练有效越狱大型模型：较低层对有害内容表现出更高的敏感性 cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.20952v1) [paper-pdf](http://arxiv.org/pdf/2502.20952v1)

**Authors**: Hongyuan Shen, Min Zheng, Jincheng Wang, Yang Zhao

**Abstract**: With the widespread application of Large Language Models across various domains, their security issues have increasingly garnered significant attention from both academic and industrial communities. This study conducts sampling and normalization of the parameters of the LLM to generate visual representations and heatmaps of parameter distributions, revealing notable discrepancies in parameter distributions among certain layers within the hidden layers. Further analysis involves calculating statistical metrics for each layer, followed by the computation of a Comprehensive Sensitivity Score based on these metrics, which identifies the lower layers as being particularly sensitive to the generation of harmful content. Based on this finding, we employ a Freeze training strategy, selectively performing Supervised Fine-Tuning only on the lower layers. Experimental results demonstrate that this method significantly reduces training duration and GPU memory consumption while maintaining a high jailbreak success rate and a high harm score, outperforming the results achieved by applying the LoRA method for SFT across all layers. Additionally, the method has been successfully extended to other open-source large models, validating its generality and effectiveness across different model architectures. Furthermore, we compare our method with ohter jailbreak method, demonstrating the superior performance of our approach. By innovatively proposing a method to statistically analyze and compare large model parameters layer by layer, this study provides new insights into the interpretability of large models. These discoveries emphasize the necessity of continuous research and the implementation of adaptive security measures in the rapidly evolving field of LLMs to prevent potential jailbreak attack risks, thereby promoting the development of more robust and secure LLMs.

摘要: 随着大型语言模型在各个领域的广泛应用，其安全问题越来越受到学术界和工业界的关注。这项研究对LLM的参数进行采样和归一化，以生成参数分布的可视化表示和热图，揭示了隐藏层内某些层之间参数分布的显著差异。进一步的分析包括计算每一层的统计指标，然后根据这些指标计算综合敏感度分数，以确定较低的层对有害内容的产生特别敏感。基于这一发现，我们采用了冻结训练策略，只在较低的层有选择地进行监督微调。实验结果表明，该方法在保持较高的越狱成功率和较高的伤害得分的同时，显著减少了训练时间和GPU内存消耗，优于将LORA方法应用于所有层的SFT的结果。此外，该方法已经成功地扩展到其他开源大型模型，验证了其跨不同模型体系结构的通用性和有效性。此外，我们将我们的方法与其他越狱方法进行了比较，证明了我们方法的优越性能。通过创新性地提出了一种逐层统计分析和比较大模型参数的方法，本研究为大模型的可解释性提供了新的见解。这些发现强调了在快速发展的LLMS领域持续研究和实施适应性安全措施的必要性，以防止潜在的越狱攻击风险，从而促进更强大和安全的LLMS的发展。



## **32. Beyond Natural Language Perplexity: Detecting Dead Code Poisoning in Code Generation Datasets**

超越自然语言困惑：检测代码生成数据集中的死代码中毒 cs.CL

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.20246v2) [paper-pdf](http://arxiv.org/pdf/2502.20246v2)

**Authors**: Chi-Chien Tsai, Chia-Mu Yu, Ying-Dar Lin, Yu-Sung Wu, Wei-Bin Lee

**Abstract**: The increasing adoption of large language models (LLMs) for code-related tasks has raised concerns about the security of their training datasets. One critical threat is dead code poisoning, where syntactically valid but functionally redundant code is injected into training data to manipulate model behavior. Such attacks can degrade the performance of neural code search systems, leading to biased or insecure code suggestions. Existing detection methods, such as token-level perplexity analysis, fail to effectively identify dead code due to the structural and contextual characteristics of programming languages. In this paper, we propose DePA (Dead Code Perplexity Analysis), a novel line-level detection and cleansing method tailored to the structural properties of code. DePA computes line-level perplexity by leveraging the contextual relationships between code lines and identifies anomalous lines by comparing their perplexity to the overall distribution within the file. Our experiments on benchmark datasets demonstrate that DePA significantly outperforms existing methods, achieving 0.14-0.19 improvement in detection F1-score and a 44-65% increase in poisoned segment localization precision. Furthermore, DePA enhances detection speed by 0.62-23x, making it practical for large-scale dataset cleansing. Overall, by addressing the unique challenges of dead code poisoning, DePA provides a robust and efficient solution for safeguarding the integrity of code generation model training datasets.

摘要: 越来越多的大型语言模型(LLM)被用于与代码相关的任务，这引起了人们对其训练数据集的安全性的担忧。一个严重的威胁是死代码中毒，即将语法上有效但功能上冗余的代码注入到训练数据中，以操纵模型行为。这种攻击可能会降低神经代码搜索系统的性能，导致代码建议有偏见或不安全。现有的检测方法，如令牌级困惑分析，由于编程语言的结构和上下文特征，无法有效地识别死代码。本文针对代码的结构特性，提出了一种新的行级检测和清理方法DEPA(Dead Code Perplexity Analyst)。DEPA通过利用代码行之间的上下文关系来计算行级困惑，并通过将它们的困惑程度与文件中的整体分布进行比较来识别异常行。我们在基准数据集上的实验表明，DEPA的性能明显优于现有的方法，在检测F1-Score上获得了0.14-0.19的改进，在有毒片段定位精度上提高了44%-65%。此外，DEPA将检测速度提高了0.62-23倍，使其适用于大规模数据集清理。总体而言，通过解决死代码中毒的独特挑战，DEPA为保护代码生成模型训练数据集的完整性提供了健壮而高效的解决方案。



## **33. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

一脚踏进门：LLC的多次越狱 cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.19820v2) [paper-pdf](http://arxiv.org/pdf/2502.19820v2)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.

摘要: 随着大型语言模型越来越多地融入现实世界的应用程序中，确保人工智能的安全至关重要。一个关键的挑战是越狱，敌意提示绕过内置的保护措施，导致有害的不允许输出。受心理学进门原则的启发，我们引入了FITD，这是一种新颖的多转弯越狱方法，它利用了这样一种现象，即较小的初始承诺降低了对更重大或更不道德的违法行为的抵抗力。我们的方法通过中间桥提示逐步升级用户查询的恶意意图，并使模型本身的响应保持一致，以诱导有毒响应。在两个越狱基准上的广泛实验结果表明，FITD在七个广泛使用的模型上实现了94%的平均攻击成功率，性能优于现有的最先进方法。此外，我们还提供了对LLM自我腐败的深入分析，强调了当前调整策略中的漏洞，并强调了多轮交互中固有的风险。代码可在https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.上获得



## **34. FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**

FLTrojan：通过选择性权重篡改对联邦语言模型进行隐私泄露攻击 cs.CR

20 pages (including bibliography and Appendix), Submitted to ACM CCS  '24

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2310.16152v3) [paper-pdf](http://arxiv.org/pdf/2310.16152v3)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) has become a key component in various language modeling applications such as machine translation, next-word prediction, and medical record analysis. These applications are trained on datasets from many FL participants that often include privacy-sensitive data, such as healthcare records, phone/credit card numbers, login credentials, etc. Although FL enables computation without necessitating clients to share their raw data, determining the extent of privacy leakage in federated language models is challenging and not straightforward. Moreover, existing attacks aim to extract data regardless of how sensitive or naive it is. To fill this research gap, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated large language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other users in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 71% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.

摘要: 联合学习(FL)已经成为机器翻译、下一词预测和病历分析等各种语言建模应用中的关键组件。这些应用程序是在来自许多FL参与者的数据集上进行训练的，这些数据集通常包括隐私敏感数据，如医疗记录、电话/信用卡号码、登录凭据等。尽管FL可以在不需要客户共享其原始数据的情况下进行计算，但在联合语言模型中确定隐私泄漏的程度是具有挑战性的，而且不是直接的。此外，现有的攻击旨在提取数据，无论它是多么敏感或幼稚。为了填补这一研究空白，我们介绍了关于从联合大型语言模型泄露隐私敏感用户数据的两个新发现。首先，我们做了一个关键的观察，在FL的中间轮中的模型快照比最终训练的模型会导致更大的隐私泄露。其次，我们发现，通过篡改模型的选择性权重可能会加剧隐私泄露，这些选择性权重专门负责记忆敏感的训练数据。我们展示了恶意客户端如何在没有任何服务器合作的情况下泄露FL中其他用户的隐私敏感数据。该方法的成员关系推理召回率提高了29%，私有数据重构效率高达71%，明显优于对敌方能力假设更强的现有攻击。



## **35. Backdooring Vision-Language Models with Out-Of-Distribution Data**

利用非分布数据进行后备视觉语言模型 cs.CV

ICLR 2025

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2410.01264v2) [paper-pdf](http://arxiv.org/pdf/2410.01264v2)

**Authors**: Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun, Lingjie Yi, Lijie Hu, Haibin Ling, Chao Chen

**Abstract**: The emergence of Vision-Language Models (VLMs) represents a significant advancement in integrating computer vision with Large Language Models (LLMs) to generate detailed text descriptions from visual inputs. Despite their growing importance, the security of VLMs, particularly against backdoor attacks, is under explored. Moreover, prior works often assume attackers have access to the original training data, which is often unrealistic. In this paper, we address a more practical and challenging scenario where attackers must rely solely on Out-Of-Distribution (OOD) data. We introduce VLOOD (Backdooring Vision-Language Models with Out-of-Distribution Data), a novel approach with two key contributions: (1) demonstrating backdoor attacks on VLMs in complex image-to-text tasks while minimizing degradation of the original semantics under poisoned inputs, and (2) proposing innovative techniques for backdoor injection without requiring any access to the original training data. Our evaluation on image captioning and visual question answering (VQA) tasks confirms the effectiveness of VLOOD, revealing a critical security vulnerability in VLMs and laying the foundation for future research on securing multimodal models against sophisticated threats.

摘要: 视觉语言模型(VLMS)的出现代表了将计算机视觉与大型语言模型(LLM)相结合以从视觉输入生成详细的文本描述方面的重大进步。尽管它们的重要性与日俱增，但VLM的安全性，特别是针对后门攻击的安全性，仍处于探索之中。此外，以前的工作通常假设攻击者可以访问原始训练数据，这通常是不现实的。在本文中，我们将讨论一种更实用、更具挑战性的场景，在该场景中，攻击者必须完全依赖分发外(OOD)数据。我们介绍了VLOOD(Backdoors Vision-Language Models with Out-Of-Distributed Data)，这是一种新的方法，具有两个关键贡献：(1)展示了在复杂的图像到文本任务中对VLM的后门攻击，同时最小化了有毒输入下原始语义的退化；(2)提出了创新的后门注入技术，而不需要访问原始训练数据。我们对图像字幕和视觉问答(VQA)任务的评估证实了VLOOD的有效性，揭示了VLMS中的一个关键安全漏洞，并为未来保护多模式模型免受复杂威胁的研究奠定了基础。



## **36. SQL Injection Jailbreak: A Structural Disaster of Large Language Models**

SQL注入越狱：大型语言模型的结构灾难 cs.CR

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2411.01565v5) [paper-pdf](http://arxiv.org/pdf/2411.01565v5)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak methods and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. For open-source models, SIJ achieves near 100\% attack success rates on five well-known LLMs on the AdvBench and HEx-PHI, while incurring lower time costs compared to previous methods. For closed-source models, SIJ achieves an average attack success rate over 85\% across five models in the GPT and Doubao series. Additionally, SIJ exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.

摘要: 近年来，大型语言模型的快速发展为各个领域带来了新的活力，产生了可观的社会效益和经济效益。然而，越狱是一种攻击形式，通过精心制作的提示诱使LLM产生有害内容，这对LLM的安全和可信开发构成了重大挑战。以前的越狱方法主要利用LLMS的内部属性或功能，例如基于优化的越狱方法和利用模型的上下文学习能力的方法。本文介绍了一种新的越狱方法--SQL注入越狱(SIJ)，它针对LLMS的外部属性，特别是LLMS构造输入提示的方式。通过在用户提示中注入越狱信息，SIJ成功地诱导该模型输出有害内容。对于开源模型，SIJ在AdvBtch和十六进制PHI上的五个知名LLM上实现了近100%的攻击成功率，同时与以前的方法相比产生了更低的时间成本。对于闭源模型，SIJ在GPT和豆宝系列中的五个模型上实现了85%以上的平均攻击成功率。此外，SIJ还暴露了LLMS中一个迫切需要缓解的新漏洞。针对这一问题，我们提出了一种简单的防御方法，称为自我提醒密钥来对抗SIJ，并通过实验结果证明了其有效性。我们的代码可以在https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.上找到



## **37. LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis**

LLM有节奏：使用令牌间时间和网络流量分析对大型语言模型进行指纹识别 cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20589v1) [paper-pdf](http://arxiv.org/pdf/2502.20589v1)

**Authors**: Saeif Alhazbi, Ahmed Mohamed Hussain, Gabriele Oligeri, Panos Papadimitratos

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into many technological ecosystems across various domains and industries, identifying which model is deployed or being interacted with is critical for the security and trustworthiness of the systems. Current verification methods typically rely on analyzing the generated output to determine the source model. However, these techniques are susceptible to adversarial attacks, operate in a post-hoc manner, and may require access to the model weights to inject a verifiable fingerprint. In this paper, we propose a novel passive and non-invasive fingerprinting technique that operates in real-time and remains effective even under encrypted network traffic conditions. Our method leverages the intrinsic autoregressive generation nature of language models, which generate text one token at a time based on all previously generated tokens, creating a unique temporal pattern like a rhythm or heartbeat that persists even when the output is streamed over a network. We find that measuring the Inter-Token Times (ITTs)-time intervals between consecutive tokens-can identify different language models with high accuracy. We develop a Deep Learning (DL) pipeline to capture these timing patterns using network traffic analysis and evaluate it on 16 Small Language Models (SLMs) and 10 proprietary LLMs across different deployment scenarios, including local host machine (GPU/CPU), Local Area Network (LAN), Remote Network, and Virtual Private Network (VPN). The experimental results confirm that our proposed technique is effective and maintains high accuracy even when tested in different network conditions. This work opens a new avenue for model identification in real-world scenarios and contributes to more secure and trustworthy language model deployment.

摘要: 随着大型语言模型(LLM)越来越多地集成到不同领域和行业的许多技术生态系统中，识别部署了哪个模型或正在与哪个模型交互对于系统的安全性和可信性至关重要。当前的验证方法通常依赖于分析生成的输出来确定源模型。然而，这些技术容易受到敌意攻击，以后自组织方式操作，并且可能需要访问模型权重才能注入可验证的指纹。在本文中，我们提出了一种新的被动和非侵入性的指纹识别技术，该技术实时运行，即使在加密的网络流量条件下也仍然有效。我们的方法利用了语言模型固有的自回归生成特性，它基于所有先前生成的令牌一次生成一个令牌，从而创建一种独特的时间模式，如节奏或心跳，即使当输出通过网络流传输时，该模式也会持续存在。我们发现，测量标记间时间(ITTS)-连续标记之间的时间间隔-可以高精度地识别不同的语言模型。我们开发了深度学习(DL)管道，通过网络流量分析来捕获这些计时模式，并在16个小语言模型(SLM)和10个专有LLM上进行了评估，这些模型跨越不同的部署场景，包括本地主机(GPU/CPU)、局域网(LAN)、远程网络和虚拟专用网(VPN)。实验结果表明，该方法是有效的，即使在不同的网络条件下也能保持较高的准确率。这项工作为现实世界场景中的模型识别开辟了一条新的途径，并有助于更安全、更可信的语言模型部署。



## **38. Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks**

了解多语言LLM对微调攻击的脆弱性 cs.CL

15 pages, 6 figures, 7 tables

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2410.18210v2) [paper-pdf](http://arxiv.org/pdf/2410.18210v2)

**Authors**: Samuele Poppi, Zheng-Xin Yong, Yifei He, Bobbie Chern, Han Zhao, Aobo Yang, Jianfeng Chi

**Abstract**: Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.

摘要: 最近大型语言模型(LLM)的进步引发了人们对其安全性的广泛担忧。最近的工作表明，通过使用一些恶意选择的指令跟随示例，即微调攻击，可以很容易地删除LLM的安全对齐。我们进一步了解多语言LLM中的微调攻击。我们首先发现了微调攻击的跨语言泛化：使用几个恶意选择的一种语言的指令跟随示例，多语言LLM也很容易被攻破(例如，多语言LLM无法拒绝其他语言的有害提示)。基于这一发现，我们假设安全相关信息是语言不可知的，并提出了一种在模型参数空间中识别安全相关信息的新方法--安全信息本地化。通过SIL语言，我们验证了这一假设，发现在微调攻击中只改变20%的权重参数就可以破坏所有语言的安全对齐。此外，我们为替代路径假说提供了证据，证明了冻结安全相关参数为什么不能防止微调攻击，并证明了我们的攻击向量仍然可以越狱适应新语言的LLM。



## **39. Trustworthy AI-Generative Content for Intelligent Network Service: Robustness, Security, and Fairness**

用于智能网络服务的值得信赖的人工智能生成内容：稳健性、安全性和公平性 cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2405.05930v2) [paper-pdf](http://arxiv.org/pdf/2405.05930v2)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Xiang Chen, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have revolutionized content creation. High-speed next-generation communication technology is an ideal platform for providing powerful AIGC network services. At the same time, advanced AIGC techniques can also make future network services more intelligent, especially various online content generation services. However, the significant untrustworthiness concerns of current AIGC models, such as robustness, security, and fairness, greatly affect the credibility of intelligent network services, especially in ensuring secure AIGC services. This paper proposes TrustGAIN, a trustworthy AIGC framework that incorporates robust, secure, and fair network services. We first discuss the robustness to adversarial attacks faced by AIGC models in network systems and the corresponding protection issues. Subsequently, we emphasize the importance of avoiding unsafe and illegal services and ensuring the fairness of the AIGC network services. Then as a case study, we propose a novel sentiment analysis-based detection method to guide the robust detection of unsafe content in network services. We conduct our experiments on fake news, malicious code, and unsafe review datasets to represent LLM application scenarios. Our results indicate that TrustGAIN is an exploration of future networks that can support trustworthy AIGC network services.

摘要: 以大型语言模型(LLM)为代表的AI生成内容(AIGC)模型彻底改变了内容创作。高速下一代通信技术是提供强大AIGC网络服务的理想平台。同时，先进的AIGC技术也可以让未来的网络服务更加智能化，特别是各种在线内容生成服务。然而，当前AIGC模型的健壮性、安全性和公平性等严重的不可信赖性问题极大地影响了智能网服务的可信度，特别是在确保安全的AIGC服务方面。提出了一种融合了健壮、安全、公平的网络服务的可信AIGC框架TrustGAIN。我们首先讨论了AIGC模型在网络系统中所面临的对抗攻击的稳健性以及相应的保护问题。随后，我们强调了避免不安全和非法服务的重要性，确保AIGC网络服务的公平性。作为一个案例，我们提出了一种新的基于情感分析的检测方法来指导对网络服务中不安全内容的稳健检测。我们在假新闻、恶意代码和不安全评论数据集上进行了实验，以表示LLM应用场景。我们的结果表明，TrustGAIN是对未来网络的探索，可以支持可信的AIGC网络服务。



## **40. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12659v3) [paper-pdf](http://arxiv.org/pdf/2502.12659v3)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: 大型推理模型的快速发展，如OpenAI-03和DeepSeek-R1，使得复杂推理相对于非推理的大型语言模型有了显著的改进。然而，它们增强的能力，再加上DeepSeek-R1等型号的开源访问，引发了严重的安全问题，特别是它们可能被滥用的问题。在这项工作中，我们提出了这些推理模型的全面安全评估，利用已建立的安全基准来评估它们是否符合安全法规。此外，我们调查了它们对敌意攻击的敏感性，例如越狱和快速注入，以评估它们在现实世界应用程序中的健壮性。通过多方面的分析，我们发现了四个重要的发现：(1)无论是在安全基准上还是在攻击上，开源的R1型号和03-mini型号之间都存在着显著的安全差距，这表明需要在R1上做出更多的安全努力。(2)与安全对齐的基本模型相比，精炼推理模型的安全性能较差。(3)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(4)与最终答案相比，R1模型的思维过程带来了更大的安全顾虑。我们的研究为推理模型的安全含义提供了见解，并强调了在R1模型的安全性方面进一步改进的必要性，以缩小差距。



## **41. Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training**

用于学习的代币，用于取消学习的代币：通过双重用途培训减轻大型语言模型中的成员推断攻击 cs.LG

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19726v1) [paper-pdf](http://arxiv.org/pdf/2502.19726v1)

**Authors**: Toan Tran, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) have become the backbone of modern natural language processing but pose privacy concerns about leaking sensitive training data. Membership inference attacks (MIAs), which aim to infer whether a sample is included in a model's training dataset, can serve as a foundation for broader privacy threats. Existing defenses designed for traditional classification models do not account for the sequential nature of text data. As a result, they either require significant computational resources or fail to effectively mitigate privacy risks in LLMs. In this work, we propose a lightweight yet effective empirical privacy defense for protecting training data of language modeling by leveraging the token-specific characteristics. By analyzing token dynamics during training, we propose a token selection strategy that categorizes tokens into hard tokens for learning and memorized tokens for unlearning. Subsequently, our training-phase defense optimizes a novel dual-purpose token-level loss to achieve a Pareto-optimal balance between utility and privacy. Extensive experiments demonstrate that our approach not only provides strong protection against MIAs but also improves language modeling performance by around 10\% across various LLM architectures and datasets compared to the baselines.

摘要: 大型语言模型(LLM)已经成为现代自然语言处理的支柱，但也带来了对敏感训练数据泄露的隐私担忧。成员身份推断攻击(MIA)旨在推断样本是否包括在模型的训练数据集中，可以作为更广泛的隐私威胁的基础。现有的为传统分类模型设计的防御措施没有考虑到文本数据的顺序性质。因此，它们要么需要大量的计算资源，要么无法有效地缓解LLMS中的隐私风险。在这项工作中，我们提出了一种轻量级但有效的经验隐私防御，通过利用特定于标记的特征来保护语言建模的训练数据。通过分析令牌在训练过程中的动态变化，提出了一种令牌选择策略，将令牌分为硬令牌和记忆令牌，分别用于学习和遗忘。随后，我们的训练阶段防御优化了一种新的两用令牌级损失，以实现效用和隐私之间的帕累托最优平衡。大量的实验表明，我们的方法不仅提供了对MIA的强大保护，而且与基准相比，在不同的LLM体系结构和数据集上，语言建模性能提高了约10%。



## **42. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

大视觉语言模型的自我监督学习视觉编码器中的秘密后门攻击 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.18290v2) [paper-pdf](http://arxiv.org/pdf/2502.18290v2)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.

摘要: 自监督学习(SSL)视觉编码者学习高质量的图像表征，因此成为开发大型视觉语言模型(LVLMS)视觉通道的重要组成部分。由于培训这类编码器的成本很高，预先训练的编码器被广泛共享并部署到许多安全关键或具有社会意义的LVLM中。在这种实际情况下，我们揭示了一种新的后门威胁，即仅仅通过损害视觉编码器就可以在这些LVLM中诱导出显著的视觉幻觉。由于这些编码器的共享和重用，许多下游的LVLM可能会继承编码器的后门行为，导致广泛的后门。在这项工作中，我们提出了BadVision，这是第一个通过新颖的触发优化和后门学习技术来利用LVLM的SSL视觉编码器中的漏洞的方法。我们在八个基准测试中评估了BadVision在两种类型的SSL编码器和LVLM上的性能。结果表明，BadVision在保持隐蔽性的同时，以99%以上的攻击成功率有效地驱动了LVLM进入攻击者选择的幻觉，导致了77.6%的相对视觉理解错误。SOTA后门检测方法不能有效检测到我们的攻击。



## **43. M-LLM Based Video Frame Selection for Efficient Video Understanding**

基于M-LLM的视频帧选择以实现高效的视频理解 cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19680v1) [paper-pdf](http://arxiv.org/pdf/2502.19680v1)

**Authors**: Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi

**Abstract**: Recent advances in Multi-Modal Large Language Models (M-LLMs) show promising results in video reasoning. Popular Multi-Modal Large Language Model (M-LLM) frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an M-LLM, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream M-LLM may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight M-LLM -based frame selection method that adaptively select frames that are more relevant to users' queries. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a M-LLM; (ii) Temporal signal, in which multiple frames selection by prompting Large Language Model (LLM) using the captions of all frame candidates. The selected frames are then digested by a frozen downstream video M-LLM for visual reasoning and question answering. Empirical results show that the proposed M-LLM video frame selector improves the performances various downstream video Large Language Model (video-LLM) across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks.

摘要: 多模式大语言模型(M-LLMS)的最新进展表明，在视频推理方面取得了可喜的结果。流行的多模式大语言模型(M-LLM)框架通常采用朴素的均匀采样来减少输入到M-LLM的视频帧的数量，特别是对于长上下文视频。然而，它可能会在视频的某些时段失去关键的上下文，从而下游的M-LLM可能没有足够的视觉信息来回答问题。针对这一痛点，我们提出了一种基于轻量级M-LLM的框架选择方法，该方法自适应地选择与用户查询更相关的框架。为了训练提出的帧选择器，我们引入了两个监督信号(I)空间信号，其中单帧重要性通过提示M-LLM进行评分；(Ii)时间信号，其中通过提示大语言模型(LLM)使用所有帧候选的字幕来选择多帧。然后，选定的帧被冻结的下行视频M-LLM消化，以进行视觉推理和问答。实验结果表明，所提出的M-LLM视频帧选择器提高了不同下游视频大语言模型(VIDEO-LLM)跨中(ActivityNet，Next-QA)和长(EgoSchema，LongVideo)上下文视频问答基准的性能。



## **44. Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack**

通过动态视觉语言对齐攻击提高MLLM中的对抗可移植性 cs.CV

arXiv admin note: text overlap with arXiv:2403.09766

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19672v1) [paper-pdf](http://arxiv.org/pdf/2502.19672v1)

**Authors**: Chenhe Gu, Jindong Gu, Andong Hua, Yao Qin

**Abstract**: Multimodal Large Language Models (MLLMs), built upon LLMs, have recently gained attention for their capabilities in image recognition and understanding. However, while MLLMs are vulnerable to adversarial attacks, the transferability of these attacks across different models remains limited, especially under targeted attack setting. Existing methods primarily focus on vision-specific perturbations but struggle with the complex nature of vision-language modality alignment. In this work, we introduce the Dynamic Vision-Language Alignment (DynVLA) Attack, a novel approach that injects dynamic perturbations into the vision-language connector to enhance generalization across diverse vision-language alignment of different models. Our experimental results show that DynVLA significantly improves the transferability of adversarial examples across various MLLMs, including BLIP2, InstructBLIP, MiniGPT4, LLaVA, and closed-source models such as Gemini.

摘要: 基于LLM构建的多模式大型语言模型（MLLM）最近因其在图像识别和理解方面的能力而受到关注。然而，虽然MLLM容易受到对抗攻击，但这些攻击在不同模型之间的可转移性仍然有限，尤其是在有针对性的攻击环境下。现有的方法主要关注特定于视觉的干扰，但难以应对视觉语言模式对齐的复杂性质。在这项工作中，我们引入了动态视觉语言对齐（DynVLA）攻击，这是一种新颖的方法，将动态扰动注入视觉语言连接器中，以增强不同模型的不同视觉语言对齐的概括性。我们的实验结果表明，DynVLA显着提高了对抗性示例在各种MLLM之间的可移植性，包括BLIP 2、INSTBLIP、MiniGPT 4、LLaVA和Gemini等闭源模型。



## **45. H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models, Including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking**

H-CoT：劫持思想链安全推理机制越狱大型推理模型，包括OpenAI o 1/o3、DeepSeek-R1和Gemini 2.0 Flash Thinking cs.CL

Website: https://maliciouseducator.org/

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12893v2) [paper-pdf](http://arxiv.org/pdf/2502.12893v2)

**Authors**: Martin Kuo, Jianyi Zhang, Aolin Ding, Qinsi Wang, Louis DiValentin, Yujia Bao, Wei Wei, Hai Li, Yiran Chen

**Abstract**: Large Reasoning Models (LRMs) have recently extended their powerful reasoning capabilities to safety checks-using chain-of-thought reasoning to decide whether a request should be answered. While this new approach offers a promising route for balancing model utility and safety, its robustness remains underexplored. To address this gap, we introduce Malicious-Educator, a benchmark that disguises extremely dangerous or malicious requests beneath seemingly legitimate educational prompts. Our experiments reveal severe security flaws in popular commercial-grade LRMs, including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking. For instance, although OpenAI's o1 model initially maintains a high refusal rate of about 98%, subsequent model updates significantly compromise its safety; and attackers can easily extract criminal strategies from DeepSeek-R1 and Gemini 2.0 Flash Thinking without any additional tricks. To further highlight these vulnerabilities, we propose Hijacking Chain-of-Thought (H-CoT), a universal and transferable attack method that leverages the model's own displayed intermediate reasoning to jailbreak its safety reasoning mechanism. Under H-CoT, refusal rates sharply decline-dropping from 98% to below 2%-and, in some instances, even transform initially cautious tones into ones that are willing to provide harmful content. We hope these findings underscore the urgent need for more robust safety mechanisms to preserve the benefits of advanced reasoning capabilities without compromising ethical standards.

摘要: 大型推理模型(LRM)最近将其强大的推理能力扩展到安全检查-使用思想链推理来决定是否应该响应请求。虽然这一新方法为平衡模型的实用性和安全性提供了一条有希望的途径，但其稳健性仍未得到充分探索。为了弥补这一差距，我们引入了恶意教育者，这是一个基准，它将极其危险或恶意的请求掩盖在看似合法的教育提示之下。我们的实验揭示了流行的商业级LRMS存在严重的安全漏洞，包括OpenAI o1/03、DeepSeek-r1和Gemini 2.0 Flash Think。例如，尽管OpenAI的o1模型最初保持了98%左右的高拒绝率，但后续的模型更新显著损害了其安全性；攻击者可以轻松地从DeepSeek-R1和Gemini 2.0 Flash Think中提取犯罪策略，而不需要任何额外的伎俩。为了进一步突出这些漏洞，我们提出了劫持思想链(H-CoT)，这是一种通用的、可转移的攻击方法，它利用模型自己展示的中间推理来越狱其安全推理机制。在H-cot下，拒绝率大幅下降--从98%降至2%以下--在某些情况下，甚至会将最初谨慎的语气转变为愿意提供有害内容的语气。我们希望这些发现强调了迫切需要更强大的安全机制，以在不损害道德标准的情况下保留先进推理能力的好处。



## **46. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

太好了，现在写一篇关于这一点的文章：渐强多转法学硕士越狱攻击 cs.CR

Accepted at USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2404.01833v3) [paper-pdf](http://arxiv.org/pdf/2404.01833v3)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.

摘要: 大型语言模型(LLM)的受欢迎程度显著提高，并越来越多地被多个应用程序采用。这些低成本管理机构强烈反对从事非法或不道德的话题，以此作为避免造成负责任的人工智能损害的一种手段。然而，最近一系列被称为越狱的袭击试图克服这种联系。直观地说，越狱攻击的目的是缩小模型可以做的事情和它愿意做的事情之间的差距。在这篇文章中，我们介绍了一种新的越狱攻击，称为Crescendo。与现有的越狱方法不同，Cresendo是一种简单的多转弯越狱方法，它以一种看似良性的方式与模型交互。它以关于手头任务的一般提示或问题开始，然后通过逐步参考模型的答复逐步升级对话，从而导致成功越狱。我们在不同的公共系统上对Cresendo进行了评估，包括ChatGPT、Gemini Pro、Gemini-Ultra、Llama-2 70b和Llama-3 70b Chat，以及Anthropic Chat。我们的结果证明了Crescendo的强大效力，它在所有评估的模型和任务中都实现了高攻击成功率。此外，我们提出了Crescendomation，这是一个自动化的Crescendo攻击工具，并通过我们的评估展示了它对最先进的模型的有效性。Cresendomation在AdvBtch子集数据集上超过了其他最先进的越狱技术，在GPT-4和Gemini-Pro上的性能分别提高了29%-61%和49%-71%。最后，我们还展示了Cresendo越狱多模式模型的能力。



## **47. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

超越表面层面模式：针对LLM越狱攻击的敏捷驱动防御框架 cs.CR

15 pages, 12 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19041v1) [paper-pdf](http://arxiv.org/pdf/2502.19041v1)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.

摘要: 尽管统一的大型语言模型(LLM)经过了拒绝有害请求的培训，但它们仍然容易受到越狱攻击。遗憾的是，现有的方法往往只关注表层模式，而忽略了更深层次的攻击本质。其结果是，当攻击提示改变时，防御就会失败，即使潜在的“攻击本质”保持不变。为了解决这个问题，我们引入了EDDF，一个针对LLMS中越狱攻击的EDDF框架。EDDF是一种即插即用的输入过滤方法，分为两个阶段：1)离线本质数据库构建，2)在线恶意查询检测。EDDF背后的关键思想是从一组不同的已知攻击实例中提取“攻击本质”，并将其存储在脱机矢量数据库中。实验结果表明，EDDF的性能明显优于现有方法，攻击成功率降低了至少20%，突出了其对越狱攻击的卓越稳健性。



## **48. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models**

针对预训练的大型语言模型的纯标签成员推断攻击 cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18943v1) [paper-pdf](http://arxiv.org/pdf/2502.18943v1)

**Authors**: Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Membership Inference Attacks (MIAs) aim to predict whether a data sample belongs to the model's training set or not. Although prior research has extensively explored MIAs in Large Language Models (LLMs), they typically require accessing to complete output logits (\ie, \textit{logits-based attacks}), which are usually not available in practice. In this paper, we study the vulnerability of pre-trained LLMs to MIAs in the \textit{label-only setting}, where the adversary can only access generated tokens (text). We first reveal that existing label-only MIAs have minor effects in attacking pre-trained LLMs, although they are highly effective in inferring fine-tuning datasets used for personalized LLMs. We find that their failure stems from two main reasons, including better generalization and overly coarse perturbation. Specifically, due to the extensive pre-training corpora and exposing each sample only a few times, LLMs exhibit minimal robustness differences between members and non-members. This makes token-level perturbations too coarse to capture such differences.   To alleviate these problems, we propose \textbf{PETAL}: a label-only membership inference attack based on \textbf{PE}r-\textbf{T}oken sem\textbf{A}ntic simi\textbf{L}arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are `better' memorized and have smaller perplexity. We conduct extensive experiments on the WikiMIA benchmark and the more challenging MIMIR benchmark. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics on five prevalent open-source LLMs.

摘要: 成员推理攻击(MIA)的目的是预测数据样本是否属于模型的训练集。尽管先前的研究已经广泛地探索了大型语言模型(LLM)中的MIA，但它们通常需要访问以完成输出日志(即，文本{基于日志的攻击})，而这在实践中通常是不可用的。在该文中，我们研究了在仅标签设置的情况下，攻击者只能访问生成的令牌(文本)的情况下，预先训练的LLMS对MIA的脆弱性。我们首先揭示了现有的仅标签MIA在攻击预先训练的LLM方面的影响很小，尽管它们在推断用于个性化LLM的微调数据集方面非常有效。我们发现，它们的失败有两个主要原因，包括较好的泛化和过粗的扰动。具体地说，由于大量的预训练语料库和每个样本只暴露几次，LLMS在成员和非成员之间表现出最小的稳健性差异。这使得令牌级的扰动过于粗略，无法捕捉到这样的差异。为了缓解这些问题，我们提出了一种基于Textbf{PE}r-\Textbf{T}Oken Sem\Textbf{A}Ntic Simi\Textbf{L}的仅标签成员关系推理攻击。具体地说，Petal利用令牌级语义相似性来近似输出概率，并随后计算困惑。它最终基于一个共同的假设来揭示成员身份，即成员的记忆力更好，困惑程度更小。我们在WikiMIA基准和更具挑战性的Mimir基准上进行了广泛的实验。根据经验，我们的Petal在针对个性化LLM的现有仅标签攻击的扩展上性能更好，甚至在五个流行的开源LLM上的所有指标上都与其他基于Logit的高级攻击相当。



## **49. JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models**

JailBench：大型语言模型的全面中国安全评估基准 cs.CL

12 pages, 5 figures, accepted at PAKDD 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18935v1) [paper-pdf](http://arxiv.org/pdf/2502.18935v1)

**Authors**: Shuyi Liu, Simiao Cui, Haoran Bu, Yuming Shang, Xi Zhang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various applications, highlighting the urgent need for comprehensive safety evaluations. In particular, the enhanced Chinese language proficiency of LLMs, combined with the unique characteristics and complexity of Chinese expressions, has driven the emergence of Chinese-specific benchmarks for safety assessment. However, these benchmarks generally fall short in effectively exposing LLM safety vulnerabilities. To address the gap, we introduce JailBench, the first comprehensive Chinese benchmark for evaluating deep-seated vulnerabilities in LLMs, featuring a refined hierarchical safety taxonomy tailored to the Chinese context. To improve generation efficiency, we employ a novel Automatic Jailbreak Prompt Engineer (AJPE) framework for JailBench construction, which incorporates jailbreak techniques to enhance assessing effectiveness and leverages LLMs to automatically scale up the dataset through context-learning. The proposed JailBench is extensively evaluated over 13 mainstream LLMs and achieves the highest attack success rate against ChatGPT compared to existing Chinese benchmarks, underscoring its efficacy in identifying latent vulnerabilities in LLMs, as well as illustrating the substantial room for improvement in the security and trustworthiness of LLMs within the Chinese context. Our benchmark is publicly available at https://github.com/STAIR-BUPT/JailBench.

摘要: 大型语言模型(LLM)在各种应用中表现出了非凡的能力，突显了对综合安全评估的迫切需要。特别是，低成本管理系统中文水平的提高，加上中文表达的独特性和复杂性，推动了针对中文的安全评估基准的出现。然而，这些基准通常不能有效地暴露LLM安全漏洞。为了弥补这一差距，我们引入了JailBch，这是第一个全面的中国基准，用于评估低成本管理中的深层漏洞，其特点是根据中国背景定制了精细化的层次化安全分类。为了提高生成效率，我们采用了一种新颖的自动越狱提示工程师(AJPE)框架来构建JailB边，该框架结合了越狱技术来增强评估有效性，并利用LLMS通过上下文学习来自动扩大数据集的规模。建议的JailBtch在13个主流LLMS上进行了广泛的评估，与现有的中国基准相比，对ChatGPT的攻击成功率最高，突显了其在识别LLMS潜在漏洞方面的有效性，并说明了LLMS在中国背景下的安全性和可信度有很大的改进空间。我们的基准测试在https://github.com/STAIR-BUPT/JailBench.上公开提供



## **50. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

利用攻击技术防御即时注入攻击 cs.CR

9 pages

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.00459v3) [paper-pdf](http://arxiv.org/pdf/2411.00459v3)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.

摘要: 随着技术的进步，大语言模型(LLM)在各种自然语言处理(NLP)任务中取得了显著的性能，支持Microsoft Copilot等LLM集成应用程序。然而，随着LLMS的不断发展，出现了新的漏洞，特别是即时注入攻击。这些攻击欺骗LLM偏离原始输入指令，并执行注入数据内容的攻击者指令，例如检索的结果。最近的攻击方法利用LLMS的指令跟随能力和它们无法区分注入到数据内容中的指令的能力，实现了高攻击成功率(ASR)。当比较攻击和防御方法时，我们有趣地发现它们有相似的设计目标，都是诱导模型忽略不想要的指令，而是执行想要的指令。因此，我们提出了一个直观的问题：这些攻击技术是否可以用于防御目的？在本文中，我们反转了快速注入方法的意图，在以前的免训练攻击方法的基础上，通过重复攻击过程来开发新的防御方法，但使用的是原始输入指令而不是注入指令。我们的综合实验表明，我们的防御技术优于现有的免训练防御方法，取得了最先进的结果。



