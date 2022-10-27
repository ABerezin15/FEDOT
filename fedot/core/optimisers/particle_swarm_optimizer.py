import datetime
import os
from typing import Sequence, Callable
import numpy as np
from functools import partial
import random

from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory, log_to_history
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
from fedot.core.utils import fedot_project_root
from fedot.core.visualisation.opt_viz_extra import OptHistoryExtraVisualizer

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

    def _initial_population(self, evaluator: Callable):
        evaluation_result = evaluator(self.initial_individuals)
        self._update_population(evaluation_result)

    def _evolve_population(self, evaluator: Callable) -> PopulationT:
        population = self.initial_individuals
        particle_best = population
        global_best = max(population, key=lambda Individual: Individual.fitness)
        w = 0.9
        c1 = 0.5
        c2 = 0.5
        r1 = np.random.random_sample()
        r2 = np.random.random_sample()
        r3 = np.random.random_sample()
        for j, current_particle in enumerate(population):
            if r1 < w:
                velocity = self.mutation(population[j])
            else:
                velocity = population[j]
            if r2 < c1:
                position = max(evaluator(self.crossover([velocity, particle_best[j]])),
                               key=lambda Individual: Individual.fitness)
            else:
                position = velocity
            if r3 < c2:
                new_position = max(evaluator(self.crossover([global_best, position])),
                                   key=lambda Individual: Individual.fitness)
            else:
                new_position = position

            if new_position.fitness > particle_best[j].fitness:
                particle_best[j] = new_position
                population[j] = particle_best[j]
            if new_position.fitness > global_best.fitness:
                global_best = new_position
                population[j] = global_best

        return population


def results_visualization(history, composed_pipelines):
    visualiser = OptHistoryExtraVisualizer()
    visualiser.visualise_history(history)
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
        crossover_prob=0.8, mutation_prob=0.9,
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

    history = OptHistory(objective)
    history_callback = partial(log_to_history, history=history)
    optimiser.set_optimisation_callback(history_callback)

    composer = GPComposer(optimizer=optimiser, composer_requirements=requirements, history=history)
    pipelines_pso_composed = composer.compose_pipeline(data=dataset_to_compose)

    if is_visualise:
        results_visualization(composed_pipelines=pipelines_pso_composed, history=composer.history)


if __name__ == '__main__':
    full_path_train, full_path_test = get_scoring_data()
    run_pso(full_path_train, full_path_test)

