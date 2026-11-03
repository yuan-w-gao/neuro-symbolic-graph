#include <sstream>

#include "../manager.hpp"
#include "extra_shrg.hpp"

namespace shrg {

std::string SHRG_CFGToString(const SHRG &self, int index) {
    auto &label_set = Manager::manager.label_set;
    auto &cfg_rule = self.cfg_rules.at(index);
    std::ostringstream os;
    debug::Printer{nullptr, label_set}(cfg_rule, self.external_nodes.size(), os);
    os << '(' << cfg_rule.shrg_index << ')';
    return os.str();
}

} // namespace shrg
