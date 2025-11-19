from collections import deque
from typing import Dict, List, Set, Tuple

import heapq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class Skill(BaseModel):
    id: str
    name: str
    related_skill_ids: List[str] = Field(default_factory=list)


class Candidate(BaseModel):
    id: str
    name: str
    skills: List[str]
    experience_years: int = 0
    interests: List[str] = Field(default_factory=list)


class Job(BaseModel):
    id: str
    title: str
    required_skills: List[str]
    nice_to_have_skills: List[str] = Field(default_factory=list)
    description: str = ""


class Course(BaseModel):
    id: str
    name: str
    teaches_skills: List[str]
    duration_hours: int


class RankRequest(BaseModel):
    job_id: str
    candidate_ids: List[str]


class RankedCandidate(BaseModel):
    candidate_id: str
    candidate_name: str
    score: int
    matching_skills: List[str]
    missing_skills: List[str]


class StudyPlanRequest(BaseModel):
    candidate_id: str
    job_id: str


class StudyPlanStep(BaseModel):
    course_id: str
    course_name: str
    skills_covered: List[str]
    duration_hours: int


class StudyPlanResponse(BaseModel):
    candidate_id: str
    job_id: str
    missing_skills: List[str]
    plan: List[StudyPlanStep]
    uncovered_skills: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    job_id: str
    title: str
    reason: str


class RecommendationResponse(BaseModel):
    candidate_id: str
    recommendations: List[Recommendation]


class Database:
    """Simple in-memory hash-table backed storage using Python dicts."""

    def __init__(self) -> None:
        self.candidates: Dict[str, Candidate] = {}
        self.jobs: Dict[str, Job] = {}
        self.skills: Dict[str, Skill] = {}
        self.courses: Dict[str, Course] = {}

    def add_candidate(self, candidate: Candidate) -> None:
        self.candidates[candidate.id] = candidate

    def add_job(self, job: Job) -> None:
        self.jobs[job.id] = job

    def add_skill(self, skill: Skill) -> None:
        self.skills[skill.id] = skill

    def add_course(self, course: Course) -> None:
        self.courses[course.id] = course

    def get_candidate(self, candidate_id: str) -> Candidate:
        try:
            return self.candidates[candidate_id]
        except KeyError as err:
            raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found") from err

    def get_job(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError as err:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from err

    def get_course(self, course_id: str) -> Course:
        try:
            return self.courses[course_id]
        except KeyError as err:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found") from err

    def list_candidates(self) -> List[Candidate]:
        return list(self.candidates.values())

    def list_jobs(self) -> List[Job]:
        return list(self.jobs.values())

    def list_courses(self) -> List[Course]:
        return list(self.courses.values())


class NetworkGraph:
    """Graph connecting candidates, skills, and jobs."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.graph: Dict[str, Set[str]] = {}
        self._build_graph()

    def _add_edge(self, src: str, dest: str) -> None:
        self.graph.setdefault(src, set()).add(dest)
        self.graph.setdefault(dest, set()).add(src)

    def _build_graph(self) -> None:
        for candidate in self.db.candidates.values():
            candidate_node = f"candidate:{candidate.id}"
            for skill_id in candidate.skills:
                if skill_id in self.db.skills:
                    skill_node = f"skill:{skill_id}"
                    # Edges here allow BFS/DFS to jump from a candidate to owned skills.
                    self._add_edge(candidate_node, skill_node)

        for skill in self.db.skills.values():
            skill_node = f"skill:{skill.id}"
            for related_skill_id in skill.related_skill_ids:
                if related_skill_id in self.db.skills:
                    related_node = f"skill:{related_skill_id}"
                    # Linking skills enlarges the neighborhood to discover "próximas" oportunidades.
                    self._add_edge(skill_node, related_node)

        for job in self.db.jobs.values():
            job_node = f"job:{job.id}"
            for skill_id in job.required_skills + job.nice_to_have_skills:
                if skill_id in self.db.skills:
                    skill_node = f"skill:{skill_id}"
                    # Skills connect to jobs that demand them, enabling indirect recommendations.
                    self._add_edge(skill_node, job_node)

    def recommend_jobs(self, candidate_id: str, limit: int = 3, max_depth: int = 4) -> List[Recommendation]:
        candidate = self.db.get_candidate(candidate_id)
        start_node = f"candidate:{candidate.id}"
        if start_node not in self.graph:
            return []

        visited = set([start_node])
        queue: deque[Tuple[str, int, List[str]]] = deque()
        queue.append((start_node, 0, []))
        candidate_skill_set = set(candidate.skills)
        recommendations: List[Recommendation] = []

        while queue and len(recommendations) < limit:
            node, depth, path = queue.popleft()
            if depth > max_depth:
                continue

            if node.startswith("job:"):
                job_id = node.split(":", 1)[1]
                job = self.db.jobs[job_id]
                missing = set(job.required_skills) - candidate_skill_set
                if missing:
                    reason = (
                        f"Gap in skills {sorted(missing)} discovered via path: "
                        f"{' -> '.join(p for p in path if p)}"
                    )
                    recommendations.append(
                        Recommendation(job_id=job.id, title=job.title, reason=reason)
                    )
                    continue

            for neighbor in self.graph.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1, path + [node]))

        return recommendations[:limit]


def rank_candidates(job: Job, candidates: List[Candidate], top_k: int = 3) -> List[RankedCandidate]:
    # heapq implements a min-heap, so we invert scores to simulate a max-heap ranking.
    heap: List[Tuple[int, str, RankedCandidate]] = []
    required = set(job.required_skills)

    for candidate in candidates:
        candidate_skill_set = set(candidate.skills)
        matching = sorted(candidate_skill_set & required)
        missing = sorted(required - candidate_skill_set)
        score = len(matching)
        ranked_candidate = RankedCandidate(
            candidate_id=candidate.id,
            candidate_name=candidate.name,
            score=score,
            matching_skills=matching,
            missing_skills=missing,
        )
        entry = (-score, candidate.id, ranked_candidate)
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        else:
            heapq.heappushpop(heap, entry)

    ordered = []
    while heap:
        ordered.append(heapq.heappop(heap)[2])
    return list(reversed(ordered))


def greedy_study_plan(candidate: Candidate, job: Job, courses: List[Course]) -> StudyPlanResponse:
    missing_skills = list(set(job.required_skills) - set(candidate.skills))
    if not missing_skills:
        return StudyPlanResponse(
            candidate_id=candidate.id,
            job_id=job.id,
            missing_skills=[],
            plan=[],
            uncovered_skills=[],
        )

    remaining = set(missing_skills)
    available_courses = courses.copy()
    plan: List[StudyPlanStep] = []

    # Greedy step: always pick the course covering most uncovered skills.
    while remaining and available_courses:
        best_course = None
        best_cover: Set[str] = set()
        for course in available_courses:
            covers = remaining & set(course.teaches_skills)
            if len(covers) > len(best_cover):
                best_course = course
                best_cover = covers
            elif len(covers) == len(best_cover) and covers:
                if best_course is None or course.duration_hours < best_course.duration_hours:
                    best_course = course
                    best_cover = covers

        if not best_course or not best_cover:
            break

        plan.append(
            StudyPlanStep(
                course_id=best_course.id,
                course_name=best_course.name,
                skills_covered=sorted(best_cover),
                duration_hours=best_course.duration_hours,
            )
        )
        remaining -= best_cover
        available_courses = [course for course in available_courses if course.id != best_course.id]

    return StudyPlanResponse(
        candidate_id=candidate.id,
        job_id=job.id,
        missing_skills=sorted(missing_skills),
        plan=plan,
        uncovered_skills=sorted(remaining),
    )


db = Database()


def seed_data() -> None:
    skills = [
        Skill(id="python", name="Python", related_skill_ids=["fastapi", "data_analysis"]),
        Skill(id="fastapi", name="FastAPI", related_skill_ids=["python", "cloud"]),
        Skill(id="data_analysis", name="Data Analysis", related_skill_ids=["python", "machine_learning"]),
        Skill(id="machine_learning", name="Machine Learning", related_skill_ids=["data_analysis", "data_engineering"]),
        Skill(id="sql", name="SQL", related_skill_ids=["data_engineering"]),
        Skill(id="cloud", name="Cloud", related_skill_ids=["devops", "fastapi"]),
        Skill(id="devops", name="DevOps", related_skill_ids=["cloud"]),
        Skill(id="data_engineering", name="Data Engineering", related_skill_ids=["sql", "machine_learning"]),
    ]

    for skill in skills:
        db.add_skill(skill)

    candidates = [
        Candidate(
            id="cand-1",
            name="Ana Torres",
            skills=["python", "fastapi", "sql"],
            experience_years=4,
            interests=["backend", "apis"],
        ),
        Candidate(
            id="cand-2",
            name="Bruno Lima",
            skills=["python", "data_analysis", "machine_learning"],
            experience_years=3,
            interests=["dados", "ia"],
        ),
        Candidate(
            id="cand-3",
            name="Camila Souza",
            skills=["python", "cloud", "devops"],
            experience_years=5,
            interests=["infra", "sre"],
        ),
        Candidate(
            id="cand-4",
            name="Diego Ramos",
            skills=["sql", "data_engineering", "python"],
            experience_years=6,
            interests=["etl", "dados"],
        ),
    ]

    for candidate in candidates:
        db.add_candidate(candidate)

    jobs = [
        Job(
            id="job-1",
            title="Backend Engineer",
            required_skills=["python", "fastapi", "sql"],
            nice_to_have_skills=["cloud"],
            description="Constrói APIs escaláveis para clientes NextStep AI.",
        ),
        Job(
            id="job-2",
            title="Machine Learning Analyst",
            required_skills=["python", "machine_learning", "data_analysis"],
            nice_to_have_skills=["sql"],
            description="Modelagem e experimentos com dados de recrutamento.",
        ),
        Job(
            id="job-3",
            title="Cloud DevOps Specialist",
            required_skills=["cloud", "devops", "python"],
            nice_to_have_skills=["fastapi"],
            description="Mantém infraestrutura multi-cloud.",
        ),
        Job(
            id="job-4",
            title="Data Platform Engineer",
            required_skills=["data_engineering", "sql", "python"],
            nice_to_have_skills=["machine_learning"],
            description="Constrói pipelines para recomendação e matching.",
        ),
    ]

    for job in jobs:
        db.add_job(job)

    courses = [
        Course(
            id="course-1",
            name="FastAPI Bootcamp",
            teaches_skills=["fastapi", "python"],
            duration_hours=20,
        ),
        Course(
            id="course-2",
            name="SQL Deep Dive",
            teaches_skills=["sql", "data_engineering"],
            duration_hours=25,
        ),
        Course(
            id="course-3",
            name="ML Foundations",
            teaches_skills=["machine_learning", "data_analysis"],
            duration_hours=30,
        ),
        Course(
            id="course-4",
            name="Cloud & DevOps Essentials",
            teaches_skills=["cloud", "devops"],
            duration_hours=35,
        ),
    ]

    for course in courses:
        db.add_course(course)


seed_data()
app = FastAPI(title="NextStep AI Backend", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/candidates", response_model=List[Candidate])
def list_candidates() -> List[Candidate]:
    return db.list_candidates()


@app.get("/jobs", response_model=List[Job])
def list_jobs() -> List[Job]:
    return db.list_jobs()


@app.get("/courses", response_model=List[Course])
def list_courses() -> List[Course]:
    return db.list_courses()


@app.post("/rank-candidates", response_model=List[RankedCandidate])
def rank_candidates_endpoint(payload: RankRequest) -> List[RankedCandidate]:
    job = db.get_job(payload.job_id)
    candidates = [db.get_candidate(candidate_id) for candidate_id in payload.candidate_ids]
    return rank_candidates(job, candidates)


@app.get("/recommend-jobs/{candidate_id}", response_model=RecommendationResponse)
def recommend_jobs_endpoint(candidate_id: str, limit: int = 3) -> RecommendationResponse:
    graph = NetworkGraph(db)
    recommendations = graph.recommend_jobs(candidate_id=candidate_id, limit=limit)
    return RecommendationResponse(candidate_id=candidate_id, recommendations=recommendations)


@app.post("/generate-study-plan", response_model=StudyPlanResponse)
def generate_study_plan_endpoint(payload: StudyPlanRequest) -> StudyPlanResponse:
    candidate = db.get_candidate(payload.candidate_id)
    job = db.get_job(payload.job_id)
    courses = db.list_courses()
    # Greedy selection ensures we always close the largest gap first, minimizing steps.
    return greedy_study_plan(candidate, job, courses)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
