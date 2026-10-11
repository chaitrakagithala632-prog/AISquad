from datetime import date, timedelta
from pathlib import Path
import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="TaskPilot AI API", version="1.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

class PlanRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=1200)
    hours_per_day: int = Field(default=2, ge=1, le=12)
    # None means auto-plan length. An explicit selection is always respected.
    days: Optional[int] = Field(default=None, ge=1, le=365)
    start_date: Optional[str] = None

class ReminderRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    due_at: Optional[str] = None


def duration_mentioned(goal: str) -> Optional[int]:
    text = goal.lower()
    patterns = [
        r"\b(?:in|within|over|for the next|next)\s+(\d{1,3})\s*(days?|weeks?|months?)\b",
        r"\b(\d{1,3})\s*(days?|weeks?|months?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            n = int(match.group(1)); unit = match.group(2)
            days = n * (7 if unit.startswith("week") else 30 if unit.startswith("month") else 1)
            return max(1, min(days, 365))
    return None


def detect_short_time_window(goal: str):
    """Detect an explicitly short deadline such as 'within 1 hour' or 'in 30 minutes'."""
    low = goal.lower()
    number_words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "half": 0.5}
    # Prefer a stated duration tied to completion/deadline wording, avoiding ordinary
    # phrases such as 'study 2 hours per day' unless they express the whole time window.
    patterns = [
        (r"\b(?:within|in|over the next|for the next)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\b", "numeric"),
        (r"\b(?:within|in|over the next|for the next)\s+(a|an|one|two|three|four|half)\s+(minutes?|mins?|hours?|hrs?)\b", "word"),
        (r"\b(?:i have|got|only have)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\b", "numeric"),
        (r"\b(?:i have|got|only have)\s+(a|an|one|two|three|four|half)\s+(minutes?|mins?|hours?|hrs?)\b", "word"),
    ]
    for pattern, kind in patterns:
        match = re.search(pattern, low)
        if not match:
            continue
        amount = float(match.group(1)) if kind == "numeric" else number_words[match.group(1)]
        unit = match.group(2)
        minutes = round(amount * (60 if unit.startswith(("hour", "hr")) else 1))
        minutes = max(5, min(minutes, 24 * 60))
        if minutes < 60:
            label = f"Next {minutes} minutes"
        elif minutes == 60:
            label = "Next 1 hour"
        elif minutes % 60 == 0:
            label = f"Next {minutes // 60} hours"
        else:
            label = f"Next {minutes // 60} hours {minutes % 60} minutes"
        return {"days": 1, "minutes": minutes, "label": label, "source": "instruction", "start_date": date.today().isoformat()}
    return None


def detect_time_window(goal: str):
    """Return a short time window explicitly requested in everyday wording."""
    low = goal.lower()
    # If the user says to begin tomorrow and continue for a stated number of days,
    # keep the full duration rather than accidentally shrinking it to one day.
    if re.search(r"\b(start|starting|begin|beginning)\s+(?:from\s+)?(?:today|tomorrow)\b", low) and duration_mentioned(goal):
        return None

    today_words = ["today", "tonight", "this morning", "this afternoon", "this evening"]
    tomorrow_words = ["tomorrow", "tomorrow morning", "tomorrow afternoon", "tomorrow evening", "tomorrow night"]
    # A direct today/tomorrow request takes priority over an unrelated dropdown length.
    if any(word in low for word in tomorrow_words):
        if "tomorrow morning" in low:
            label = "Tomorrow morning"
        elif "tomorrow afternoon" in low:
            label = "Tomorrow afternoon"
        elif "tomorrow evening" in low or "tomorrow night" in low:
            label = "Tomorrow evening"
        else:
            label = "Tomorrow"
        return {"days": 1, "label": label, "source": "instruction", "start_date": (date.today() + timedelta(days=1)).isoformat()}
    if any(word in low for word in today_words):
        if "tonight" in low or "this evening" in low:
            label = "Tonight"
        elif "this morning" in low:
            label = "This morning"
        elif "this afternoon" in low:
            label = "This afternoon"
        else:
            label = "Today"
        return {"days": 1, "label": label, "source": "instruction", "start_date": date.today().isoformat()}

    if re.search(r"\bthis weekend\b", low):
        remaining = (5 - date.today().weekday()) % 7 + 1 if date.today().weekday() <= 5 else 1
        remaining = max(1, min(2, remaining))
        return {"days": remaining, "label": "This weekend", "source": "instruction", "start_date": date.today().isoformat()}
    if re.search(r"\bthis week\b", low):
        remaining = 7 - date.today().weekday()
        return {"days": max(1, remaining), "label": "This week", "source": "instruction", "start_date": date.today().isoformat()}
    if re.search(r"\bnext week\b", low):
        return {"days": 7, "label": "Next week", "source": "instruction", "start_date": (date.today() + timedelta(days=(7-date.today().weekday()))).isoformat()}
    if re.search(r"\bthis month\b", low):
        tomorrow = date.today() + timedelta(days=1)
        month_end = date.today().replace(day=28) + timedelta(days=4)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        remaining = (month_end - date.today()).days + 1
        return {"days": max(1, remaining), "label": "This month", "source": "instruction", "start_date": date.today().isoformat()}
    return None


PLAIN_LANGUAGE = [
    (r"\bnon-negotiables\b", "must-haves"),
    (r"\bcontingency\b", "extra money for unexpected costs"),
    (r"\bencumbrances\b", "legal claims or debts attached to the property"),
    (r"\beasements\b", "legal rights for others to use part of the land"),
    (r"\bgeotechnical\b", "soil and ground"),
    (r"\bdeliverables\b", "work you'll receive"),
    (r"\bdefect liability\b", "period when the contractor must fix defects"),
    (r"\bbill of quantities\b", "item-by-item list of materials and costs"),
    (r"\bcross-ventilation\b", "airflow through the house"),
    (r"\bmobilisation\b", "getting the crew and equipment ready"),
    (r"\bjurisdiction\b", "local area"),
    (r"\bsnag list\b", "list of problems to fix"),
    (r"\bsnag inspection\b", "walk-through to find problems"),
    (r"\bhold points\b", "required inspection stages"),
    (r"\bmasonry\b", "brick or block work"),
    (r"\bconcealment\b", "covering it up"),
    (r"\bearthing\b", "electrical grounding"),
    (r"\bdistribution board\b", "electrical panel"),
    (r"\bsanitaryware\b", "toilets, sinks and bathroom fixtures"),
    (r"\bas-built documents\b", "final drawings showing what was actually built"),
    (r"\bdependencies\b", "steps that must happen first"),
    (r"\bre-sequence\b", "change the order of"),
    (r"\bprioritise\b", "choose what to do first"),
    (r"\bprioritised\b", "ordered by what matters most"),
    (r"\bcriteria\b", "checks"),
    (r"\bvalidate\b", "check"),
    (r"\bvalidated\b", "checked"),
    (r"\bconstraints\b", "limits"),
    (r"\bmilestones\b", "main stages"),
    (r"\breproducible\b", "easy for someone else to set up"),
    (r"\bacceptance checklist\b", "checklist to confirm it works as expected"),
]

def plain_text(value: str) -> str:
    for pattern, replacement in PLAIN_LANGUAGE:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def simplify_tasks(tasks):
    for item in tasks:
        for key in ("title", "description", "success_criteria"):
            if isinstance(item.get(key), str):
                item[key] = plain_text(item[key])
        for key in ("action_steps", "materials"):
            item[key] = [plain_text(x) for x in item.get(key, [])]
    return tasks


def choose_duration(goal: str) -> tuple[int, str]:
    low = goal.lower()
    stated = duration_mentioned(goal)
    if stated:
        return stated, "instruction"
    if any(k in low for k in ["build a house", "building a house", "construct a house", "house construction", "build my house", "construct a home"]):
        return 180, "estimated"
    if any(k in low for k in ["website", "web app", "hackathon", "software", "application", "app", "prototype"]):
        return 14, "estimated"
    if any(k in low for k in ["learn", "course", "programming", "python", "java", "skill"]):
        return 30, "estimated"
    if any(k in low for k in ["exam", "study", "revision", "gate", "test"]):
        return 14, "estimated"
    if any(k in low for k in ["routine", "habit", "fitness", "workout", "healthy"]):
        return 21, "estimated"
    return 7, "estimated"


def task(title, description, priority="High", steps=None, materials=None, success=None, estimate=60):
    return {"title": title, "description": description, "priority": priority,
            "action_steps": steps or [], "materials": materials or [],
            "success_criteria": success or "The task is complete when you can show a concrete result.",
            "estimate_minutes": estimate}


def task_blueprint(goal: str):
    low = goal.lower()
    if any(k in low for k in ["build a house", "building a house", "construct a house", "house construction", "build my house", "construct a home"]):
        return [
            task("Define needs, budget and house brief", "Turn the idea into a written brief before spending money. Estimate the house size, number of floors and rooms, preferred finish level, budget range, family needs and target move-in date.", steps=["List rooms and approximate sizes; note future needs such as parking or accessibility.", "Set a total budget range including design fees, approvals, utilities, contingency and furnishing.", "List non-negotiables versus features that can be postponed.", "Record who will make decisions and how changes will be approved."], materials=["Notebook or spreadsheet", "Budget estimate", "Household requirements"], success="A one-page brief, a budget range and a list of priorities are written down.", estimate=180),
            task("Choose and verify the plot/location", "Shortlist a suitable site based on access, services, legal status and local conditions. Do not buy or begin work based only on appearance or a seller's promise.", steps=["Compare commute, road access, drainage, water, electricity, internet and nearby services.", "Ask a local property lawyer/surveyor to verify title, boundaries, encumbrances and permitted land use.", "Check flood history, soil conditions, slope, access rights and any easements.", "Confirm local zoning/setback rules and whether the intended house is allowed before committing."], materials=["Site visits", "Land records/title documents", "Surveyor and property lawyer", "Local authority guidance"], success="The plot passes independent legal and technical checks before purchase.", estimate=480),
            task("Hire qualified local professionals", "Engage appropriately licensed/registered professionals for design, structural engineering and approvals. Requirements vary by location, so confirm the local rules before work starts.", steps=["Interview an architect/designer and review relevant completed projects.", "Engage a structural engineer and, where needed, a geotechnical/soil-testing professional.", "Agree on scope, fees, drawings, timelines, site visits and written deliverables.", "Confirm contractor references, insurance and written contract terms."], materials=["Professional references", "Scope of work", "Written quotations and contracts"], success="Responsibilities, deliverables and fees are documented and qualified professionals are appointed.", estimate=240),
            task("Survey the site and test soil", "Establish the actual site boundaries and ground conditions so the design and foundation are based on evidence rather than guesswork.", steps=["Arrange a boundary/topographic survey by a qualified surveyor.", "Arrange soil investigation as advised by the structural/geotechnical engineer.", "Check site levels, drainage direction, nearby structures and utility connection points.", "Share the reports with the design and structural team before finalising foundations."], materials=["Survey report", "Soil investigation report", "Site photographs and utility details"], success="The design team has verified measurements and ground-condition reports.", estimate=240),
            task("Create the architectural design", "Turn the brief into a functional plan that considers room flow, daylight, ventilation, privacy, safety and future maintenance.", steps=["Prepare concept layouts and compare at least two workable options.", "Review room dimensions, stairs, storage, kitchen/bathroom placement and circulation.", "Plan daylight, cross-ventilation, rainwater handling, parking and outdoor space.", "Review the design with the household and record revisions before detailed drawings."], materials=["House brief", "Site survey", "Architectural drawings", "Local design requirements"], success="The preferred layout is approved by the owner and ready for technical coordination.", estimate=480),
            task("Finalise structural and service drawings", "Coordinate engineering drawings for the structure and essential services before construction. Do not improvise structural sizes or reinforcement on site.", steps=["Have the structural engineer design foundations, columns, beams, slabs and reinforcement for the actual site and design loads.", "Plan electrical circuits, earthing, plumbing, drainage, water storage and ventilation with qualified professionals.", "Coordinate openings, ducts, shafts and service routes to avoid later cutting or rework.", "Review the drawing set for version/date and distribute only approved drawings to the site team."], materials=["Architectural drawings", "Structural drawings", "Electrical and plumbing plans", "Soil report"], success="A coordinated, dated construction drawing set is issued by the responsible professionals.", estimate=480),
            task("Get permits and written approvals", "Confirm the permits, inspections and utility permissions required by the local authority. The exact list depends on the plot and jurisdiction.", steps=["Ask the local municipal/development authority which building permission and documents are required.", "Submit the required site, ownership, architectural and engineering documents through the official process.", "Confirm any water, sewer, electricity, road-access or environmental requirements.", "Do not begin regulated construction until the required approvals are issued; keep copies on site."], materials=["Ownership documents", "Signed drawings", "Application forms and fee receipts", "Approval records"], success="Required approvals are issued and their conditions are understood before work starts.", estimate=360),
            task("Select contractor and create the work schedule", "Choose a contractor based on capability, references, transparent pricing and a written scope. A low quote alone is not proof of value.", steps=["Obtain itemised quotations based on the same approved drawings and specifications.", "Check recent projects, references, staffing, safety practices and subcontractor arrangements.", "Write down inclusions/exclusions, payment milestones, quality checks, change orders and defect liability.", "Create a sequence for site preparation, foundations, structure, services, finishes and inspections."], materials=["Approved drawings", "Bill of quantities/estimates", "Written contract", "Schedule and payment milestones"], success="A signed contract and realistic milestone schedule are in place.", estimate=360),
            task("Prepare the site and arrange safe logistics", "Prepare the site only after required approvals and contractor mobilisation. Protect neighbours, workers, access routes and the environment.", steps=["Confirm site boundaries and mark the approved building footprint with the surveyor.", "Arrange safe access, temporary storage, waste collection, water and power as permitted.", "Agree on site fencing, personal protective equipment, emergency contacts and hazard controls.", "Photograph the original site condition and nearby structures before work begins."], materials=["Approved site plan", "Survey marking", "Safety plan", "Temporary facilities and waste plan"], success="The site is safely organised and set out to the approved drawings.", estimate=240),
            task("Complete foundations with inspections", "The foundation type, depth, reinforcement and concrete specification must come from the project's qualified engineer and soil report. This checklist is for coordination, not a construction design.", steps=["Have the contractor excavate only to the approved setting-out and engineered details.", "Arrange the required engineer/authority inspection before covering reinforcement or concrete work.", "Verify materials and concrete testing/records required by the specification.", "Record inspection approvals, photographs and any corrective work before proceeding."], materials=["Approved foundation drawings", "Specified concrete and reinforcement", "Inspection records", "PPE and site safety controls"], success="Foundation work passes the required inspections and is documented before the next stage.", estimate=720),
            task("Build the structure and weatherproof shell", "Coordinate structural work, walls, roof and openings against approved drawings and inspection requirements. Never change a beam, column, slab or wall that carries load without written engineering approval.", steps=["Follow the engineer-approved sequence for columns, beams, slabs and masonry.", "Check alignment, dimensions, material deliveries and curing/quality requirements as specified.", "Install roof and external openings/details to limit water entry.", "Arrange inspections at required hold points and log defects before they are concealed."], materials=["Approved structural and architectural drawings", "Specified structural and masonry materials", "Quality/inspection log", "Safety equipment"], success="The shell is completed to approved drawings and required inspection records are signed off.", estimate=1440),
            task("Install plumbing, electrical and other services", "Use qualified tradespeople and the approved service plans. Plan testing and inspections before walls and floors are closed up.", steps=["Install water supply, drainage and ventilation routes as designed.", "Install electrical conduits, wiring, distribution board, earthing and protection as specified by the qualified electrician.", "Coordinate internet, solar, air-conditioning or other systems if included in the design.", "Test and document systems before plastering or closing access panels; correct failures first."], materials=["Approved service drawings", "Specified pipes, cables and fittings", "Qualified tradespeople", "Test/inspection records"], success="Services are tested, documented and pass required inspections before concealment.", estimate=960),
            task("Complete finishes and fittings", "Choose durable finishes that fit the budget and climate. Approve samples and measurements before ordering large quantities.", steps=["Confirm samples for flooring, paint, doors, windows, sanitaryware and cabinetry.", "Check surface preparation and moisture issues before applying finishes.", "Install floors, doors, windows, fixtures, cabinets and paint in the correct trade sequence.", "Protect completed surfaces and maintain a snag list for scratches, leaks, uneven work or incomplete items."], materials=["Approved finish schedule", "Material samples", "Measured quantities", "Delivery and defect log"], success="Finishes match approved selections and the snag list is being resolved.", estimate=1440),
            task("Inspect, test and prepare for handover", "Before moving in, verify safety, operation, approvals and documentation with the responsible professionals. Do not treat appearance alone as proof that a house is safe.", steps=["Complete a room-by-room snag inspection with the contractor and designer.", "Test plumbing for leaks, electrical protection and other installed systems with qualified tradespeople.", "Collect completion/occupancy approvals where required, warranties, manuals, drawings and test certificates.", "Agree in writing on defect corrections, final payments and maintenance responsibilities."], materials=["Snag checklist", "Test certificates", "Warranties and manuals", "Approval and as-built documents"], success="Required approvals and handover documents are collected, safety checks are complete and outstanding defects have owners/dates.", estimate=480),
        ]
    if any(k in low for k in ["exam", "study", "subject", "semester", "gate", "test", "revision"]):
        return [
            task("Collect syllabus and exam constraints", "Build the plan around the actual syllabus, exam date, marks pattern and available study time.", steps=["List every unit and subtopic from the official syllabus.", "Collect class notes, teacher-marked questions and past papers.", "Mark each topic strong, medium or weak.", "Set a daily study slot and reserve time for revision and practice."], materials=["Syllabus", "Class notes", "Past papers", "Calendar"], success="A complete topic checklist and realistic study schedule are ready."),
            task("Prioritise and schedule topics", "Order the work by exam relevance, prerequisite concepts and personal weakness rather than studying randomly.", steps=["Group topics into small 30–60 minute sessions.", "Identify high-mark and frequently tested topics without ignoring the syllabus.", "Schedule difficult topics when energy is highest.", "Leave buffer time for missed sessions and a final review."], materials=["Topic checklist", "Past papers", "Study timetable"], success="Every syllabus topic has a planned study slot."),
            task("Learn concepts actively", "Study a small topic, close the notes, then explain it in your own words with an example or diagram.", steps=["Read one concept from the class material.", "Write a short explanation from memory.", "Work through one example or diagram.", "Mark doubts and resolve them before moving on."], materials=["Textbook or notes", "Notebook", "Practice examples"], success="You can explain the topic without simply rereading the page."),
            task("Practise exam-style questions", "Turn knowledge into marks by writing answers or solving problems under realistic conditions.", steps=["Attempt questions without viewing solutions first.", "Use the expected answer format and time limit.", "Compare with a reliable solution or marking scheme.", "Record the exact reason for each lost mark."], materials=["Past papers", "Timer", "Answer key/marking scheme"], success="A set of attempted questions and an error log are completed."),
            task("Repair weak areas", "Use the error log to decide what to relearn; don't just repeat questions until the answer is memorised.", steps=["Sort mistakes into concept gap, memory gap, careless error or time-management issue.", "Relearn the underlying concept.", "Solve a different question that tests the same skill.", "Update a short revision note with the corrected method."], materials=["Error log", "Notes", "Extra practice questions"], success="Previously weak questions can be solved independently."),
            task("Take a timed mock test", "Simulate the exam to check recall, accuracy and time allocation.", steps=["Choose a representative set of questions.", "Follow the time limit and avoid notes.", "Mark the paper honestly and review every mistake.", "Adjust the remaining plan based on the result."], materials=["Mock paper", "Timer", "Marking scheme"], success="A score and a prioritised list of final revision needs are recorded."),
            task("Revise and consolidate", "Use active recall and spaced review to strengthen topics that are easy to forget.", steps=["Review short notes rather than rereading whole chapters.", "Answer quick recall questions from memory.", "Revisit errors and formulas/definitions.", "Prepare a final checklist for exam day."], materials=["Short notes", "Flashcards or recall questions", "Error log"], success="The final revision checklist is complete and the remaining weak spots are known.", priority="Medium"),
        ]
    if any(k in low for k in ["website", "web app", "hackathon", "software", "application", "app", "prototype", "build"]):
        return [
            task("Define the user problem and success criteria", "Specify who the product serves, what pain point it solves and how the demo will prove value.", steps=["Write a one-sentence problem statement.", "Describe the target user and their current workaround.", "Choose three measurable success criteria.", "Define a short demo scenario from start to finish."], materials=["Problem statement", "User scenario", "Acceptance checklist"], success="The team agrees on the user, problem, MVP and demo outcome."),
            task("Design the user journey and screens", "Map the shortest path from user input to a useful outcome before adding extra features.", steps=["Sketch the main screen and key states.", "List empty, loading, success and error states.", "Choose what information each screen must show.", "Get quick feedback from a teammate and revise."], materials=["Wireframe or paper sketch", "Feature list", "User feedback"], success="A simple screen flow and component list are ready."),
            task("Set up and verify the project", "Create a reproducible development setup and confirm the app runs before building more features.", steps=["Create clear frontend, backend and asset folders.", "Install dependencies from a requirements/package file.", "Add a health-check endpoint or basic smoke test.", "Record the exact run commands in README."], materials=["Laptop", "Code editor", "Git repository", "Dependency files"], success="A teammate can run the starter project using the written instructions."),
            task("Build the core feature end-to-end", "Implement one complete working user flow before polishing optional features.", steps=["Connect the input form to the backend endpoint.", "Validate inputs and return a predictable structured response.", "Render the response clearly in the UI.", "Show useful loading and error messages."], materials=["API contract", "Frontend components", "Backend route", "Test input"], success="The main user journey works from input to visible result."),
            task("Add detailed actionable outputs", "Make generated results useful enough that a user knows what to do next, what is needed and how to verify completion.", steps=["For each task include a plain-language explanation.", "Add ordered action steps and needed materials/tools where relevant.", "Define completion criteria and a realistic time estimate.", "Include safety, legal or professional checks for high-impact activities."], materials=["Task schema", "Example goals", "Review checklist"], success="A generated task contains clear steps, resources and a definition of done."),
            task("Test and fix realistic failures", "Test more than the happy path so the demo doesn't break when a judge tries a different input.", steps=["Test empty, very short and long instructions.", "Test explicit duration and automatic duration.", "Check that returned tasks match the requested plan length.", "Test refresh, mobile layout and API failure messaging."], materials=["Test checklist", "Browser dev tools", "API docs"], success="Critical tests pass and remaining limitations are documented."),
            task("Polish the demo and explain differentiation", "Show the action loop: instruction → plan → execution → progress → adaptation, rather than presenting a chat answer alone.", steps=["Prepare a 60–120 second demo scenario.", "Show tasks with steps and completion criteria.", "Demonstrate marking a task done and viewing progress.", "Explain what is currently prototype logic and what would require a real LLM/integrations."], materials=["Demo script", "Seed scenario", "README", "Known-limitations list"], success="The demo is repeatable and claims accurately match implemented features.", priority="Medium"),
        ]
    if any(k in low for k in ["learn", "course", "programming", "python", "java", "skill"]):
        return [
            task("Define the skill outcome", "Choose a practical outcome that can be demonstrated, not just a list of videos to watch.", steps=["Write what you want to be able to build or explain.", "Check prerequisites and choose one beginner-friendly learning source.", "Set up the required tools and a place to save notes.", "Choose a tiny project that will prove the skill."], materials=["Learning roadmap", "Laptop", "Notes folder"], success="A specific skill target and proof project are selected."),
            task("Learn the first core concept", "Study one concept and immediately test whether you understand it.", steps=["Read/watch one focused lesson.", "Write a short explanation in your own words.", "Run or work through the example yourself.", "Change one input and predict what happens before testing."], materials=["Learning source", "Code editor or practice space", "Notebook"], success="You can explain and reproduce the example independently."),
            task("Practise with small exercises", "Build fluency by solving exercises without copying a finished solution.", steps=["Attempt three small problems.", "Test normal and edge cases.", "Read errors carefully and fix the cause.", "Save a clean final version and note what you learned."], materials=["Practice problems", "Editor", "Test cases"], success="Three exercises pass your own tests."),
            task("Build a mini-project", "Apply several concepts to one small usable result.", steps=["Break the project into input, processing and output.", "Build the smallest working version first.", "Test it with realistic inputs.", "Write a README explaining how to run it and what you learned."], materials=["Project brief", "Editor", "Version control if available"], success="A runnable mini-project and README are ready."),
            task("Review and demonstrate the skill", "Use recall and a small demonstration to identify gaps before moving to harder topics.", steps=["Explain the main concepts without notes.", "Rebuild one part from scratch.", "List unresolved questions and answer them.", "Choose the next project-sized challenge."], materials=["Project", "Notes", "Self-review checklist"], success="You can demonstrate the skill and identify the next learning step.", priority="Medium"),
        ]
    return [
        task("Clarify the desired outcome", f"Define what success looks like for: {goal.strip().rstrip('.')}.", steps=["Describe the final result in one sentence.", "List constraints such as budget, time, people and quality.", "Choose a measurable sign of completion.", "Break the result into major milestones."], materials=["Goal statement", "Calendar", "Budget or constraints list"], success="The outcome, constraints and success measure are written down."),
        task("Gather resources and verify requirements", "Collect the information, materials, tools and people needed before committing time or money.", steps=["List required and optional resources.", "Identify missing information and who can confirm it.", "Compare reliable options, costs and dependencies.", "Flag any safety, legal or expert approvals needed."], materials=["Resource checklist", "Quotes or source information", "Contact list"], success="The critical requirements and resource gaps are known."),
        task("Create a practical step-by-step plan", "Turn the goal into milestones and order them according to dependencies.", steps=["List the major milestones from start to finish.", "Identify which steps depend on earlier decisions.", "Assign owners and realistic time estimates.", "Add a buffer for delays and a way to track progress."], materials=["Calendar or planner", "Milestone checklist", "Time estimates"], success="Each milestone has an order, owner or next action."),
        task("Complete the first high-impact action", "Start with a small, concrete action that reduces uncertainty or unlocks later work.", steps=["Choose the highest-impact available step.", "Gather only what is needed to begin.", "Complete the action and record the result.", "Update the plan if new information changes the next step."], materials=["Task checklist", "Required tools or documents"], success="A concrete deliverable or decision is completed."),
        task("Check progress and remove blockers", "Compare actual progress with the plan and address the biggest obstacle first.", steps=["Review completed work and open tasks.", "Identify the main delay or quality gap.", "Ask for help or obtain missing information if needed.", "Re-sequence the next tasks when dependencies change."], materials=["Progress log", "Blocker list", "Relevant contacts"], success="The main blocker has an owner and a next action."),
        task("Validate the result", "Check that the output meets the original success criteria, not just that work was performed.", steps=["Compare the result with the requirements.", "Test or inspect important parts.", "Fix the highest-impact defects.", "Ask a relevant person to review if the result affects others."], materials=["Original requirements", "Review checklist", "Test/inspection tools"], success="The result has been checked against the agreed criteria."),
        task("Review, document and plan the next cycle", "Capture what worked and what remains so the process improves rather than restarting from zero.", steps=["Record the outcome and lessons learned.", "List unfinished work and why it remains.", "Update estimates based on actual effort.", "Choose the next milestone or maintenance check."], materials=["Progress notes", "Updated checklist", "Calendar"], success="The outcome and next actions are documented.", priority="Medium"),
    ]


def make_tasks(goal: str, days: int, hours: int, window_minutes: Optional[int] = None):
    blueprint = task_blueprint(goal)
    if window_minutes is not None:
        # Short deadlines should produce a small, doable set of actions that fit the
        # actual available time, not a full-day or multi-day plan.
        low = goal.lower()
        if any(k in low for k in ["build a house", "building a house", "construct a house", "house construction", "construct a home"]):
            short_blueprint = [blueprint[0]]
        elif any(k in low for k in ["exam", "study", "revision", "gate", "test"]):
            short_blueprint = [
                task("Pick the most important topics", "Use the time you have to focus on the topics most likely to matter. Do not try to reread everything.", steps=["Check your syllabus, assignment brief or teacher's important questions.", "Choose the two or three topics you find hardest or most important.", "Keep your notes and practice questions ready before you begin."], materials=["Syllabus or task instructions", "Notes", "Practice questions"], success="You have a short list of the most important things to work on.", estimate=10),
                task("Work on the highest-priority topic", "Study or practise one important topic actively so you can remember and use it, rather than only reading it.", steps=["Review the key idea for a few minutes.", "Close your notes and explain the idea in your own words.", "Solve one or two questions without looking at the answer.", "Check mistakes and correct them."], materials=["Notes", "Pen and paper", "Practice questions"], success="You can explain the idea and complete at least one practice question.", estimate=25),
                task("Do a quick final check", "Use the remaining time to find gaps and decide what needs attention next.", steps=["Try a short question or recall the main points from memory.", "Check your answer and correct any mistake.", "Write down anything you still need to review."], materials=["Practice question", "Quick checklist"], success="You have checked your understanding and noted any remaining gaps.", estimate=10),
            ]
        elif any(k in low for k in ["assignment", "report", "essay", "homework", "presentation"]):
            short_blueprint = [
                task("Check exactly what must be submitted", "Quickly confirm the instructions and identify the minimum work needed to submit a complete, sensible version on time.", steps=["Read the question or marking rubric once.", "List the required sections or questions.", "Mark the most important part and note any file or formatting rules."], materials=["Assignment instructions", "Rubric", "Laptop or notebook"], success="You know what must be included and what to work on first.", estimate=8),
                task("Complete the most important section first", "Spend most of your available time producing the main answer or core content. Aim for clear and correct before polishing.", steps=["Start with the question or section worth the most marks.", "Write a simple first draft using your own words.", "Add the evidence, examples, calculations or references the instructions require."], materials=["Assignment file", "Class notes or approved sources"], success="The main part of the assignment is written and saved.", estimate=35),
                task("Check and submit-ready the work", "Use the remaining time to fix obvious gaps and make sure the file can be submitted.", steps=["Check that every required question or section is present.", "Correct obvious spelling, calculation or formatting mistakes.", "Save the file with a clear name and confirm the submission method."], materials=["Submission checklist", "Required file format"], success="The work is saved, checked and ready to submit, or the remaining gap is clearly identified.", estimate=12),
            ]
        elif any(k in low for k in ["website", "web app", "hackathon", "software", "application", "app", "prototype", "code"]):
            short_blueprint = [
                task("Choose the one outcome to finish first", "With a short deadline, focus on one useful result instead of trying to build every feature.", steps=["Write down what the user must be able to do.", "Choose the smallest working feature that proves the idea.", "Keep optional design changes and extra features for later."], materials=["Project brief", "Laptop", "Code editor"], success="One clear, small outcome is selected.", estimate=8),
                task("Build the smallest working version", "Implement the core path first and keep the code simple. Test it as soon as it works.", steps=["Create or open the needed files.", "Build the main screen or core function.", "Connect only the pieces required for the main action.", "Run it and fix the first blocking error."], materials=["Laptop", "Code editor", "Existing project files"], success="The main action works at least once from start to finish.", estimate=35),
                task("Test the main action and note next steps", "Check the core feature from a user's point of view and record anything still broken.", steps=["Try the main action with a realistic example.", "Check what happens when input is empty or incorrect.", "Fix the most visible problem and write down remaining work."], materials=["Test example", "Short checklist"], success="The main feature has been tested and the next improvement is clear.", estimate=12),
            ]
        else:
            short_blueprint = [
                task("Clarify the result you need", "Turn your instruction into one clear result that can realistically be worked on in the time available.", steps=["Write what must be finished before time runs out.", "Separate must-do items from optional extras.", "Gather the information or tools needed to start."], materials=["Your instructions", "A notes app or paper", "Required tools"], success="The required result and first action are clear.", estimate=8),
                task("Work on the most important action", "Spend most of the available time doing the work itself instead of over-planning.", steps=["Start the highest-priority part.", "Work in small steps and save your progress.", "If you get stuck, note the blocker and move to a useful part you can complete."], materials=["Tools and information needed for the task"], success="The main piece of work is completed or meaningfully advanced.", estimate=35),
                task("Review what is done", "Use the last few minutes to check the result and decide whether anything essential is missing.", steps=["Compare your result with the original instruction.", "Fix the most important gap.", "Save or share the result and note any unfinished item honestly."], materials=["Original instruction", "Quick checklist"], success="The result is checked and saved, with remaining work clearly noted.", estimate=12),
            ]
        count = 1 if window_minutes < 20 else 2 if window_minutes < 40 else 3
        chosen = short_blueprint[:count]
        # Scale estimates to fit the user's time limit, leaving a little breathing room.
        raw_total = sum(max(1, x["estimate_minutes"]) for x in chosen)
        budget = max(5, window_minutes - min(5, max(1, window_minutes // 10)))
        remaining = budget
        selected = []
        for idx, item in enumerate(chosen):
            left = len(chosen) - idx
            estimate = max(1, min(item["estimate_minutes"], remaining // left))
            selected.append((1, item, estimate))
            remaining -= estimate
    elif days == 1:
        # A one-day request should show a small, realistic set of actions for that day,
        # not only the first step of a much larger project.
        count = min(len(blueprint), max(2, min(4, hours + 1)))
        chosen = blueprint[:count]
        per_task_minutes = max(15, (hours * 60) // max(1, len(chosen)))
        selected = [(1, item, per_task_minutes) for item in chosen]
    elif days <= len(blueprint):
        selected = [(i + 1, blueprint[i], max(15, min(blueprint[i]["estimate_minutes"], hours * 60))) for i in range(days)]
    else:
        selected = [(round(i * (days - 1) / (len(blueprint) - 1)) + 1, item,
                     max(15, min(item["estimate_minutes"], hours * 60)))
                    for i, item in enumerate(blueprint)] if len(blueprint) > 1 else [(1, blueprint[0], max(15, min(blueprint[0]["estimate_minutes"], hours * 60)))]
    tasks = []
    for index, (day_number, base, estimate) in enumerate(selected):
        tasks.append({"id": index + 1, "title": base["title"], "description": base["description"],
                      "action_steps": base["action_steps"], "materials": base["materials"],
                      "success_criteria": base["success_criteria"], "priority": base["priority"],
                      "day": day_number, "estimate_minutes": estimate, "completed": False})
    return simplify_tasks(tasks)

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "TaskPilot AI", "mode": "structured-planner-prototype"}

@app.post("/api/plan")
def create_plan(request: PlanRequest):
    goal = request.goal.strip()
    if len(goal) < 3:
        raise HTTPException(status_code=400, detail="Please describe a goal in a little more detail.")
    try:
        short_window = detect_short_time_window(goal)
        window = detect_time_window(goal) if short_window is None else None
        window_minutes = short_window["minutes"] if short_window else None
        if short_window:
            days, duration_source = 1, short_window["source"]
            schedule_window = short_window["label"]
            start_date = short_window["start_date"]
        elif window:
            days, duration_source = window["days"], window["source"]
            schedule_window = window["label"]
            start_date = window["start_date"]
        elif request.days is None:
            days, duration_source = choose_duration(goal)
            schedule_window = None
            starts_tomorrow = bool(re.search(r"\b(start|starting|begin|beginning)\s+(?:from\s+)?tomorrow\b", goal.lower()))
            start_date = (date.today() + timedelta(days=1)).isoformat() if starts_tomorrow else date.today().isoformat()
        else:
            days, duration_source = request.days, "selected"
            schedule_window = None
            starts_tomorrow = bool(re.search(r"\b(start|starting|begin|beginning)\s+(?:from\s+)?tomorrow\b", goal.lower()))
            start_date = (date.today() + timedelta(days=1)).isoformat() if starts_tomorrow else date.today().isoformat()
        tasks = make_tasks(goal, days, request.hours_per_day, window_minutes=window_minutes)
        title = goal[:64] + ("…" if len(goal) > 64 else "")
        notes = []
        if "house" in goal.lower() or "construction" in goal.lower():
            notes.append("House construction timelines depend on location, approvals, design, weather, contractor availability and site conditions. This is a planning checklist, not engineering or legal advice; use qualified local professionals and required approvals.")
        if days > 30 and duration_source == "selected":
            notes.append("Your selected duration is preserved. If this timeline is shorter than the real-world work requires, the plan is a compressed schedule, not a guarantee of completion.")
        if window_minutes is not None and any(k in goal.lower() for k in ["house", "construction", "website", "web app", "application", "software", "learn", "course"]):
            notes.append("This short timeframe is not enough to finish a large project. This plan focuses on the most useful actions you can complete in the time available; it does not promise the whole project will be finished.")
        return {"planTitle": title, "goal": goal, "days": days, "durationSource": duration_source,
                "scheduleWindow": schedule_window, "windowMinutes": window_minutes, "startDate": start_date,
                "hoursPerDay": request.hours_per_day, "tasks": tasks, "plannerMode": "structured-demo",
                "note": "TaskPilot uses everyday wording, clear steps and a short plan for the time window you requested. This prototype uses structured planning rules rather than a connected LLM.",
                "cautions": notes}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not create plan: {exc}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
