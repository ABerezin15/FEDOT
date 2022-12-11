import datetime
from typing import Sequence, Callable
import numpy as np
import random
import os

from fedot.core.repository.tasks import Task
from fedot.core.composer.gp_composer.gp_composer import GPComposer
from fedot.core.optimisers.populational_optimizer import PopulationalOptimizer
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.optimisers.gp_comp.operators.crossover import Crossover, CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import Mutation, MutationTypesEnum
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.optimizer import GraphGenerationParams
from cases.credit_scoring.credit_scoring_problem import get_scoring_data
from fedot.core.visualisation.opt_viz_extra import OptHistoryExtraVisualizer
from fedot.core.utils import fedot_project_root

random.seed(12)
np.random.seed(12)


class ParticleSwarmOptimizer(PopulationalOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPGraphOptimizerParameters):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)

        self.crossover = Crossover(graph_optimizer_params, requirements, graph_generation_params)
        self.mutation = Mutation(graph_optimizer_params, requirements, graph_generation_params)
        self.initial_individuals = [Individual(graph) for graph in initial_graphs]
        self.w = 0.9  # inertia weight
        self.c1 = 0.5  # acceleration coefficient 1
        self.c2 = 0.5  # acceleration coefficient 2
        self.r1 = np.random.random_sample()  # random number on interval [0, 1)
        self.r2 = np.random.random_sample()  # random number on interval [0, 1)
        self.r3 = np.random.random_sample()  # random number on interval [0, 1)

    def _initial_population(self, evaluator: Callable):
        evaluation_result = evaluator(self.initial_individuals)
        self._update_population(evaluation_result)

    def _evolve_population(self, evaluator: Callable) -> PopulationT:

        """In this method the particle velocities and positions are updated by three operations.
        The first operation 'velocity', which represents the velocity operation for a particle using mutation process
        which is applied with a probability of w and generates a temporary particle. The second operation 'position'
        which is the cognitive part of particle represents the crossover operation, which is applied with a probability
        of c1. The third operation 'new_position', which is the social part of particle represents the crossover
        operator, which is applied with a probability of c2."""
        
        population = self.initial_individuals
        particle_best = population
        global_best = max(population, key=lambda Individual: Individual.fitness)
        for j, current_particle in enumerate(population):
            if self.r1 < self.w:
                velocity = self.mutation(population[j])
            else:
                velocity = population[j]
            if self.r2 < self.c1:
                position = max(evaluator(self.crossover([velocity, particle_best[j]])),
                               key=lambda Individual: Individual.fitness)
            else:
                position = velocity
            if self.r3 < self.c2:
                new_position = max(evaluator(self.crossover([position, global_best])),
                                   key=lambda Individual: Individual.fitness)
            else:
                new_position = position

            if new_position.fitness > particle_best[j].fitness:
                particle_best[j] = new_position

            if new_position.fitness > global_best.fitness:
                global_best = new_position

        return population


def results_visualization(history, composed_pipelines):
    visualiser = OptHistoryExtraVisualizer()
    visualiser.visualise_history(history)
    history.show()
    composed_pipelines.show()


def run_pso(train_file_path, test_file_path,
            timeout: datetime.timedelta = datetime.timedelta(minutes=20),
            is_visualise = True):
    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    available_model_types = get_operations_for_task(task=task, mode='model')

    requirements = PipelineComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types,
        timeout=timeout,
        num_of_generations=20
    )

    optimiser_parameters = GPGraphOptimizerParameters(
        pop_size=20,
        crossover_prob=1, mutation_prob=1,
        mutation_types=[MutationTypesEnum.simple],
        crossover_types=[CrossoverTypesEnum.one_point])

    graph_generation_params = get_pipeline_generation_params(requirements)

    objective = Objective([ClassificationMetricsEnum.ROCAUC])

    initial = [random_graph(graph_generation_params=graph_generation_params,
                            requirements=requirements) for i in range(10)]

    optimiser = ParticleSwarmOptimizer(
        graph_generation_params=graph_generation_params,
        objective=objective,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements, initial_graphs=initial)


    composer = GPComposer(optimizer=optimiser, composer_requirements=requirements)
    pipelines_pso_composed = composer.compose_pipeline(data=dataset_to_compose)

    if is_visualise:
        results_visualization(composed_pipelines=pipelines_pso_composed, history=composer.history)


if __name__ == '__main__':
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)
    run_pso(full_path_train, full_path_test)
