// Stands in for a Python Worker where the Host's bridge would be: fyflow's runtime takes only JS Workers, and the
// declarations test needs none to run.
export default class PythonStandIn {
  run(): never {
    throw new Error("A Python Worker runs only in a fyflow-app Host");
  }
}
