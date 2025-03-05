#include <rocRoller/Expression.hpp>
namespace rocRollerTest {
/**
         * @brief Widen (u)int32 to (u)int64.
         *
         * Has many assumptions in input expr. See the implementation for details.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
	rocRoller::Expression::ExpressionPtr widenTo64bit(rocRoller::Expression::ExpressionPtr expr);
}
